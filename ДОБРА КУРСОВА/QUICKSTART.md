# 🚀 Швидкий старт

## Встановлення базових залежностей

Якщо повне встановлення з `requirements.txt` займає багато часу, встановіть спочатку базові пакети:

```bash
pip install python-dotenv nltk scikit-learn matplotlib seaborn plotly gradio pandas numpy joblib
```

## Встановлення для роботи з ембедингами (опціонально)

Для notebooks 03-06 потрібні додаткові пакети (можуть займати час):

```bash
pip install sentence-transformers transformers umap-learn
```

## Встановлення NLTK даних

Після встановлення nltk, завантажте необхідні дані:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Структура виконання

### Мінімальний старт (без ML моделей):

1. `01_data_cleaning.ipynb` - очищення даних
2. `02_eda_preprocessing.ipynb` - аналіз даних

### Повний процес (потребує встановлення всіх залежностей):

1. `01_data_cleaning.ipynb`
2. `02_eda_preprocessing.ipynb`
3. `03_embeddings_viz.ipynb` (потребує sentence-transformers)
4. `04_training.ipynb`
5. `05_evaluation.ipynb`
6. `06_post_analysis.ipynb`

### Запуск Gradio demo:

```bash
python demo/app.py
```

**Примітка:** Потребує навченої моделі з notebook 04.

## Альтернативний підхід

Якщо torch встановлюється занадто довго, можете використовувати CPU версію:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers umap-learn
```

## Перевірка встановлення

```bash
python -c "import gradio, pandas, sklearn, matplotlib; print('All core packages installed!')"
```

## Проблеми та рішення

### ModuleNotFoundError

Встановіть відсутній пакет:

```bash
pip install <package-name>
```

### Помилки з NLTK

```python
import nltk
nltk.download('all')  # Завантажити всі дані
```

### Повільне встановлення torch

- Використовуйте CPU версію (див. вище)
- Або встановіть без torch і використовуйте альтернативні моделі
