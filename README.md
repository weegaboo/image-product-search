# Image Product Search

Сервис для поиска похожих товаров по фотографии с использованием CLIP + FAISS. Загрузи фото — получи top-K наиболее похожих товаров из каталога. Поддерживает добавление и удаление изображений, создание и удаление товаров. Имеется простой UI на Streamlit.

## 🔧 Быстрый запуск

```bash
docker compose up --build -d
```

API: http://localhost:8000/docs  
UI: http://localhost:8501

## 🧪 Альтернативный запуск (без Docker)

```bash
uv pip install --system
uvicorn search.application:app --reload
streamlit run ui/application.py
```

## 📁 Структура

- `products/` — изображения товаров  
- `search/` — FastAPI backend  
- `ui/` — Streamlit UI  
- `pyproject.toml`, `uv.lock` — зависимости (через [uv](https://github.com/astral-sh/uv))  
- `docker-compose.yml` — запуск всего проекта

## 🖼️ Формат изображений

Фотографии товаров должны быть организованы по папкам с именами `product_id`. Каждая папка содержит изображения, относящиеся к одному товару. Формат файлов — `.jpg`, `.jpeg` или `.png`.

Пример:

```
products/
├── 100012345678/
│   ├── front.jpg
│   ├── side.jpg
│   └── top.png
├── 100098765432/
│   └── photo1.jpg
```

После запуска сервиса все изображения автоматически индексируются и становятся доступны для поиска.

---

Python • FastAPI • Streamlit • CLIP • FAISS
