import os
import shutil
import uuid

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from io import BytesIO
from fastapi.staticfiles import StaticFiles


# Настройка модели CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Хранилища
DATA_DIR = "./products"
os.makedirs(DATA_DIR, exist_ok=True)

products = {}  # product_id: List[image_paths]
product_id_map = []  # Индексы в FAISS -> product_id
index = faiss.IndexFlatL2(512)  # FAISS index


# Функция генерации эмбеддинга
def embed_image(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding /= embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")


# Стартовая индексация
def build_index_from_folder():
    global index, product_id_map
    index.reset()
    product_id_map.clear()
    products.clear()

    for product_id in tqdm(os.listdir(DATA_DIR)):
        product_dir = os.path.join(DATA_DIR, product_id)
        if not os.path.isdir(product_dir):
            continue
        products[product_id] = []
        for image_file in os.listdir(product_dir):
            image_path = os.path.join(product_dir, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                embedding = embed_image(image)
                index.add(embedding)
                product_id_map.append(product_id)
                products[product_id].append(image_path)
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")


app = FastAPI()
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")


@app.on_event("startup")
def on_startup():
    build_index_from_folder()


# Ручка: добавить новый товар
@app.post("/add_product/")
def add_product():
    product_id = str(uuid.uuid4())
    os.makedirs(os.path.join(DATA_DIR, product_id), exist_ok=True)
    products[product_id] = []
    return {"product_id": product_id}


# Ручка: добавить изображение к товару
@app.post("/add_image/{product_id}")
async def add_image(product_id: str, file: UploadFile = File(...)):
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    product_path = os.path.join(DATA_DIR, product_id)
    os.makedirs(product_path, exist_ok=True)

    image_name = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(product_path, image_name)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    image = Image.open(image_path).convert("RGB")
    embedding = embed_image(image)
    index.add(embedding)
    product_id_map.append(product_id)
    products[product_id].append(image_path)

    return {"message": "Image added", "path": image_path}


# Ручка: удалить изображение у товара
@app.delete("/delete_image/{product_id}")
def delete_image(product_id: str, filename: str):
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    image_path = os.path.join(DATA_DIR, product_id, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    os.remove(image_path)
    build_index_from_folder()  # простая реализация
    return {"message": "Image deleted"}


# Ручка: удалить товар полностью
@app.delete("/delete_product/{product_id}")
def delete_product(product_id: str):
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    shutil.rmtree(os.path.join(DATA_DIR, product_id))
    build_index_from_folder()
    return {"message": "Product deleted"}


# Ручка: поиск топ-k похожих товаров
@app.post("/search/")
async def search(file: UploadFile = File(...), k: int = 5):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    query_embedding = embed_image(image)

    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Index is empty")

    distances, indices = index.search(query_embedding, k)

    seen = set()
    results = []
    for i in indices[0]:
        product_id = product_id_map[i]
        if product_id in seen:
            continue
        seen.add(product_id)

        photo_paths = products.get(product_id, [])
        # возвращаем относительные пути, чтобы стримлит мог запросить через /static
        relative_paths = [
            os.path.relpath(p, DATA_DIR).replace("\\", "/") for p in photo_paths
        ]
        results.append({"product_id": product_id, "photos": relative_paths})

    return {"matches": results}
