import os
import shutil
import uuid
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
import open_clip
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-H-14",
    pretrained="laion2b_s32b_b79k",
    device=DEVICE
)
model.eval()


DATA_DIR = "./products"
os.makedirs(DATA_DIR, exist_ok=True)
products: dict[str, list[str]] = {}  # product_id -> [image_path, …]


VECTOR_SIZE = 1024
COLLECTION  = "products_clip"
client = QdrantClient(  # если контейнер крутится на других хосте/порте, поправьте
    host="localhost",
    port=6333,
    prefer_grpc=True,   # быстрее
)

def ensure_collection():
    if COLLECTION in {c.name for c in client.get_collections().collections}:
        return
    client.create_collection(
        COLLECTION,
        vectors_config=qdrant.VectorParams(
            size=VECTOR_SIZE,
            distance=qdrant.Distance.COSINE,
        ),
    )

ensure_collection()


def embed_image(image: Image.Image) -> np.ndarray:
    """
    Принимает PIL‑картинку, возвращает L2‑нормированный вектор float32 (1, 1024).
    """
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast(DEVICE == "cuda"):
        emb = model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True) # L2
    return emb.cpu().numpy().astype("float32")

def point_id() -> str:
    # Qdrant может принимать int или str. Берём str‑uuid для простоты.
    return str(uuid.uuid4())

def build_index_from_folder():
    """
    1) Чистим/пересоздаём коллекцию.
    2) Проходимся по каталогу и шлем батчами по 64 изображения.
    """
    ensure_collection()
    client.delete_collection(COLLECTION)
    ensure_collection()

    products.clear()

    batch_vectors, batch_ids, batch_payloads = [], [], []
    BATCH = 64

    for product_id in tqdm(os.listdir(DATA_DIR), desc="Scanning folder"):
        product_dir = os.path.join(DATA_DIR, product_id)
        if not os.path.isdir(product_dir):
            continue
        products[product_id] = []
        for img_file in os.listdir(product_dir):
            img_path = os.path.join(product_dir, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                vec = embed_image(image)[0]          # (1024,)
                pid = point_id()

                batch_vectors.append(vec)
                batch_ids.append(pid)
                batch_payloads.append(
                    {
                        "product_id": product_id,
                        "image_path": img_path,
                    }
                )
                products[product_id].append(img_path)
            except Exception as e:
                print(f"Ошибка {img_path}: {e}")

            # batched upsert
            if len(batch_ids) >= BATCH:
                client.upsert(
                    COLLECTION,
                    [
                        qdrant.PointStruct(id=i, vector=v, payload=p)
                        for i, v, p in zip(batch_ids, batch_vectors, batch_payloads)
                    ],
                )
                batch_vectors, batch_ids, batch_payloads = [], [], []

    # хвост
    if batch_ids:
        client.upsert(
            COLLECTION,
            [
                qdrant.PointStruct(id=i, vector=v, payload=p)
                for i, v, p in zip(batch_ids, batch_vectors, batch_payloads)
            ],
        )

app = FastAPI()
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

@app.on_event("startup")
def _startup():
    build_index_from_folder()

@app.post("/add_product/")
def add_product():
    product_id = str(uuid.uuid4())
    os.makedirs(os.path.join(DATA_DIR, product_id), exist_ok=True)
    products[product_id] = []
    return {"product_id": product_id}

@app.post("/add_image/{product_id}")
async def add_image(product_id: str, file: UploadFile = File(...)):
    if product_id not in products:
        raise HTTPException(404, "Product not found")

    product_dir = os.path.join(DATA_DIR, product_id)
    os.makedirs(product_dir, exist_ok=True)

    fname = f"{uuid.uuid4().hex}_{file.filename}"
    img_path = os.path.join(product_dir, fname)

    with open(img_path, "wb") as f:
        f.write(await file.read())

    image = Image.open(img_path).convert("RGB")
    vec = embed_image(image)[0]

    client.upsert(
        COLLECTION,
        [
            qdrant.PointStruct(
                id=point_id(),
                vector=vec,
                payload={
                    "product_id": product_id,
                    "image_path": img_path,
                },
            )
        ],
    )
    products[product_id].append(img_path)
    return {"message": "Image added", "path": img_path}

@app.delete("/delete_image/{product_id}")
def delete_image(product_id: str, filename: str):
    """
    Простейшая реализация: удаляем файл, затем строим коллекцию заново.
    Можно оптимизировать через фильтр payload'ов и точечный delete_points.
    """
    if product_id not in products:
        raise HTTPException(404, "Product not found")

    img_path = os.path.join(DATA_DIR, product_id, filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image not found")

    os.remove(img_path)
    build_index_from_folder()
    return {"message": "Image deleted"}

@app.delete("/delete_product/{product_id}")
def delete_product(product_id: str):
    if product_id not in products:
        raise HTTPException(404, "Product not found")
    shutil.rmtree(os.path.join(DATA_DIR, product_id))
    build_index_from_folder()
    return {"message": "Product deleted"}

# ──────────────────────────────────────────────────────────────────────────────
# 6.2  Поиск ближайших товаров
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/search/")
async def search(file: UploadFile = File(...), k: int = 5):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    query_vec = embed_image(image)[0].tolist()

    # ищем с запасом, чтобы потом сгруппировать по product_id
    limit = k * 10
    hits = client.query_points(
        COLLECTION,
        query=query_vec,
        limit=limit
    )

    if not hits:
        raise HTTPException(400, "Index is empty")

    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for p in hits.points:
        dist = p.score  # для cosine similarity: чем БОЛЬШЕ, тем ближе
        pid = p.payload["product_id"]
        img = p.payload["image_path"]
        grouped[pid].append((dist, img))

    # сортируем товары по максимальной близости (‑score↓)
    ranked = sorted(grouped.items(), key=lambda x: max(d for d, _ in x[1]), reverse=True)

    results = []
    for pid, items in ranked[:k]:
        # берём до 5 лучших фото внутри товара
        top_imgs = sorted(items, key=lambda x: x[0], reverse=True)[:5]
        results.append(
            {
                "product_id": pid,
                "photos": [str(Path(p).relative_to(DATA_DIR)) for _, p in top_imgs],
            }
        )

    return {"matches": results}