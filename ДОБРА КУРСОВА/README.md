# 🔍 Fake News Classifier

Проект для класифікації фейкових новин за допомогою машинного навчання та трансформерних ембедингів.

## 📁 Структура проекту

```
fake-news-classifier/
├── src/                      # Основна логіка проекту
│   ├── config.py            # Конфігурація та константи
│   ├── data_loader.py       # Завантаження датасетів
│   ├── preprocessing.py     # Препроцесинг тексту
│   ├── eda.py              # Exploratory Data Analysis
│   ├── embeddings.py       # Генерація ембедингів
│   ├── dimensionality.py   # Зменшення розмірності
│   ├── train.py            # Навчання моделей
│   ├── evaluate.py         # Оцінка моделей
│   └── post_analysis.py    # Пост-аналіз
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_embeddings_viz.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_post_analysis.ipynb
├── demo/                    # Gradio демо
│   └── app.py
├── data/                    # Дані
│   ├── raw/                # Оригінальні датасети
│   └── processed/          # Оброблені дані
├── models/                  # Збережені моделі
├── outputs/                 # Графіки та звіти
├── .env.example            # Шаблон змінних оточення
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Швидкий старт

### 1. Клонування репозиторію

```bash
git clone <repository-url>
cd fake-news-classifier
```

### 2. Створення віртуального середовища

```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

### 3. Встановлення залежностей

**Базова установка (рекомендовано для старту):**

```bash
pip install -r requirements.txt
```

**Для роботи з ML моделями (notebooks 03-06):**

```bash
pip install sentence-transformers transformers umap-learn
```

**Якщо PyTorch встановлюється занадто довго, використовуйте CPU версію:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers umap-learn
```

**Завантаження NLTK даних:**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

Детальніше про встановлення див. у [QUICKSTART.md](QUICKSTART.md)

### 4. Налаштування змінних оточення

```bash
cp .env.example .env
# Відредагуйте .env за потреби
```

### 5. Завантаження даних

Помістіть ваші CSV файли з новинами в папку `data/raw/`

## 📊 Використання

### Jupyter Notebooks

Виконайте notebooks у порядку:

1. **01_data_cleaning.ipynb** - Завантаження та очищення даних
2. **02_eda_preprocessing.ipynb** - EDA та препроцесинг
3. **03_embeddings_viz.ipynb** - Генерація ембедингів та візуалізація
4. **04_training.ipynb** - Навчання моделей
5. **05_evaluation.ipynb** - Оцінка результатів
6. **06_post_analysis.ipynb** - Додатковий аналіз

### Python scripts

```python
from src.data_loader import load_all_datasets
from src.preprocessing import preprocess_dataframe
from src.embeddings import TextEmbedder
from src.train import train_model_with_gridsearch

# Завантаження даних
df = load_all_datasets()

# Препроцесинг
df = preprocess_dataframe(df, text_column='text')

# Генерація ембедингів
embedder = TextEmbedder()
embeddings = embedder.encode_dataframe(df)

# Навчання моделі
from src.train import split_data
X_train, X_test, y_train, y_test = split_data(embeddings, df['label'])
model = train_model_with_gridsearch(X_train, y_train, model_name='logistic_regression')
```

### Запуск Gradio Demo

```bash
python demo/app.py
```

Потім відкрийте браузер за адресою `http://localhost:7860`

## 🔧 Конфігурація

Основні параметри в `.env`:

- `MODEL_NAME` - Назва sentence-transformer моделі
- `RANDOM_STATE` - Seed для відтворюваності
- `TEST_SIZE` - Розмір тестової вибірки
- `BATCH_SIZE` - Розмір батчу для ембедингів

## 📈 Моделі

Підтримувані моделі:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Всі моделі навчаються з GridSearchCV для оптимізації гіперпараметрів.

## 🧪 Тестування

```bash
pytest tests/
```

## 📝 Ліцензія

MIT License

## 👥 Автори

Команда 1 - Курсова робота

## 🤝 Контрибуція

Pull requests вітаються! Для великих змін спочатку відкрийте issue.
