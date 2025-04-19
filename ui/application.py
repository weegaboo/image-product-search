import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
from urllib.parse import quote

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Product Matcher", layout="wide")
st.title("🧠 Image Product Matcher")

# TAB layout
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Поиск", "➕ Добавить фото", "➕ Новый товар", "❌ Удалить"]
)

# -------------------------------
# TAB 1 — ПОИСК
# -------------------------------
with tab1:
    st.header("🔍 Найти похожие товары по изображению")
    search_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    k = st.slider("Сколько результатов показать", 1, 10, 5)

    if search_file and st.button("Найти"):
        uploaded_bytes = search_file.read()

        st.subheader("🔎 Вы загрузили:")
        st.image(Image.open(BytesIO(uploaded_bytes)), caption="Запрос", width=300)

        files = {"file": (search_file.name, uploaded_bytes, search_file.type)}
        response = requests.post(f"{API_URL}/search/", files=files, params={"k": k})

        if response.status_code == 200:
            results = response.json().get("matches", [])

            if not results:
                st.warning("Ничего не найдено.")
            else:
                st.subheader("🎯 Найденные товары:")

                # Стилизация карточек
                # Стилизация карточек
                st.markdown(
                    """
                    <style>
                    .product-gallery {
                        display: flex;
                        overflow-x: auto;
                        gap: 24px;
                        padding: 1em 0;
                    }
                    .product-card {
                        flex: 0 0 auto;
                        padding: 10px;
                        border-radius: 8px;
                        border: 1px solid #ddd;
                        background: #fafafa;
                        text-align: center;
                        width: 240px;
                    }
                    .product-card h4 {
                        margin-bottom: 0.5em;
                        font-size: 16px;
                        color: #333;
                    }
                    .product-card img {
                        max-width: 100%;
                        max-height: 150px;
                        margin-bottom: 6px;
                        border-radius: 4px;
                        object-fit: contain;
                    }
                    </style>
                """,
                    unsafe_allow_html=True,
                )

                # Сборка всех карточек в одну строку
                gallery_html = '<div class="product-gallery">'

                for match in results:
                    product_id = match["product_id"]
                    photos = match.get("photos", [])

                    st.markdown(f"### 🧾 Product ID: `{product_id}`")

                    cols = st.columns(min(3, len(photos)))
                    for i, photo_path in enumerate(photos[:3]):
                        try:
                            safe_path = quote(photo_path)
                            url = f"{API_URL}/static/{safe_path}"
                            response = requests.get(url)
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                with cols[i]:
                                    st.image(
                                        img,
                                        caption=photo_path.split("/")[-1],
                                        width=300,
                                    )
                            else:
                                st.error(
                                    f"Ошибка загрузки {url} ({response.status_code})"
                                )
                        except Exception as e:
                            st.error(f"Ошибка: {e}")
        else:
            st.error(f"Ошибка: {response.status_code} — {response.text}")


# -------------------------------
# TAB 2 — ДОБАВИТЬ ФОТО
# -------------------------------
with tab2:
    st.header("➕ Добавить изображение к товару")
    add_photo_product_id = st.text_input("ID товара", key="add_photo_pid")
    add_photo_file = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png"], key="add_photo_file"
    )

    if st.button("Добавить изображение") and add_photo_file and add_photo_product_id:
        files = {
            "file": (add_photo_file.name, add_photo_file.read(), add_photo_file.type)
        }
        resp = requests.post(f"{API_URL}/add_image/{add_photo_product_id}", files=files)
        if resp.status_code == 200:
            st.success("Фото успешно добавлено")
        else:
            st.error(f"Ошибка: {resp.status_code} — {resp.text}")


# -------------------------------
# TAB 3 — ДОБАВИТЬ ТОВАР
# -------------------------------
with tab3:
    st.header("➕ Создать новый товар")
    if st.button("Создать"):
        resp = requests.post(f"{API_URL}/add_product/")
        if resp.status_code == 200:
            new_id = resp.json().get("product_id")
            st.success(f"Новый товар создан: `{new_id}`")
        else:
            st.error(f"Ошибка: {resp.status_code} — {resp.text}")


# -------------------------------
# TAB 4 — УДАЛЕНИЕ
# -------------------------------
with tab4:
    st.header("❌ Удалить")

    st.markdown("##### Удалить изображение у товара")
    del_photo_pid = st.text_input("ID товара", key="del_img_pid")
    del_photo_filename = st.text_input("Имя файла", key="del_img_filename")

    if st.button("Удалить изображение"):
        resp = requests.delete(
            f"{API_URL}/delete_image/{del_photo_pid}",
            params={"filename": del_photo_filename},
        )
        if resp.status_code == 200:
            st.success("Изображение удалено")
        else:
            st.error(f"Ошибка: {resp.status_code} — {resp.text}")

    st.markdown("##### Удалить товар полностью")
    del_product_id = st.text_input("ID товара", key="del_product")

    if st.button("Удалить товар"):
        resp = requests.delete(f"{API_URL}/delete_product/{del_product_id}")
        if resp.status_code == 200:
            st.success("Товар удалён")
        else:
            st.error(f"Ошибка: {resp.status_code} — {resp.text}")
