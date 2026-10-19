# AI4SEC spam detector

Антиспам-классификатор писем (TF-IDF + LinearSVC с калибровкой) и классификатор категорий.
Команда AI4SEC (А. Белогорцев, М. Горбунов), НИУ ВШЭ МИЭМ, курс «Применение ML в кибербезопасности».

- `GET /health` — состояние и порог
- `POST /predict` `{"text": "..."}` — вердикт, вероятность спама, категория
- `POST /predict_batch` `{"texts": ["...", "..."]}`
- `GET /` — демо-страница

Запуск локально: `docker build -t spam-detector . && docker run -p 8000:8000 spam-detector`
