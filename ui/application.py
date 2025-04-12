import os

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Product Matcher", layout="wide")
st.title("Product Matcher UI")

tab1, tab2 = st.tabs(["🔍 Поиск по фото", "➕ Добавить фото к товару"])

# === TAB 1: ПОИСК ===
with tab1:
    st.header("Поиск похожих товаров по изображению")
    search_file = st.file_uploader(
        "Загрузите изображение для поиска", type=["jpg", "jpeg", "png"]
    )
    k = st.slider("Количество результатов", 1, 10, 5)

    if st.button("Найти похожие товары") and search_file:
        files = {"file": (search_file.name, search_file.read(), search_file.type)}
        response = requests.post(f"{API_URL}/search/", files=files, params={"k": k})

        if response.status_code == 200:
            result = response.json()
            print(result)
            matches = result.get("matches", [])
            st.subheader("Результаты:")

            cols = st.columns(3)
            for i, match in enumerate(matches):
                product_id = match["product_id"]
                photos = match.get("photos", [])
                with cols[i % 3]:
                    st.markdown(f"**Product ID:** `{product_id}`")
                    for path in photos:
                        image_response = requests.get(f"{API_URL}/static/{path}")
                        if image_response.status_code == 200:
                            img = Image.open(BytesIO(image_response.content))
                            st.image(img, use_column_width=True)
        else:
            st.error(f"Ошибка: {response.status_code} — {response.text}")


# === TAB 2: ДОБАВИТЬ ФОТО ===
with tab2:
    st.header("Добавить изображение к товару")
    product_id = st.text_input("ID товара")
    new_photo = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png"], key="upload"
    )

    if st.button("Добавить фото") and product_id and new_photo:
        files = {"file": (new_photo.name, new_photo.read(), new_photo.type)}
        response = requests.post(f"{API_URL}/add_image/{product_id}", files=files)

        if response.status_code == 200:
            st.success("Фото успешно добавлено")
        else:
            st.error(f"Ошибка: {response.status_code} — {response.text}")
