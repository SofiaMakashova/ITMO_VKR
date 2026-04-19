# Прогнозирование финансовых временных рядов с применением методов причинно-следственного вывода

Репозиторий для хранения данных, скриптов и результатов выпускной квалификационной работы (ВКР).  
Цель — улучшение прогнозирования финансовых временных рядов за счёт выявления и использования каузальных (причинно-следственных) связей между факторами.

## Структура репозитория
`├── datasets/ # Исходные данные для экспериментов`

`├── figures/ # Визуализации (графики, DAG-схемы, сравнения моделей)`

`├── responses/ # Отзывы научных руководителей`

`├── results/ # Результаты экспериментов (JSON, Excel)`

`├── synthetic_data/ # Синтетические данные для тестирования методов`

`├── run_pipeline.py # Запуск полного пайплайна`

`├── step0_DE/ # Сбор данных в папку datasets`

`├── step1_dag_definitions.py`

`├── step2_variable_selection.py`

`├── step3_causal_models.py`

`├── step4_baseline_models.py`

`├── step5_comparison_report.py`

`├── synthetic_causal_data.ipynb`

`└── requirements.txt`

## Возможности

- Построение причинно-следственных графов (DAG)
- Отбор переменных на основе каузальных критериев
- Обучение каузальных моделей (например, Double ML, Causal Forest)
- Сравнение с baseline-моделями (ARIMA, LSTM, XGBoost)
- Генерация синтетических данных с известной причинной структурой
- Автоматическая генерация отчётов и сохранение результатов

## Технологии

- Python 3.9+
- pandas, numpy
- causal-learn / dowhy / econml (каузальный вывод)
- scikit-learn, statsmodels
- matplotlib, seaborn
- openpyxl (для Excel)

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/SofiaMakashova/ITMO_VKR.git
cd ваш-репозиторий
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск отдельных этапов
```bash
python step1_dag_definitions.py
python step2_variable_selection.py
python step3_causal_models.py
python step4_baseline_models.py
python step5_comparison_report.py
```

### 4. Запуск всего пайплайна (рекомендованный способ)
```bash
python run_pipeline.py
```
После выполнения результаты появятся в папке results/, графики — в figures/.


# Автор
Макашова С.А.
[[Ссылка на профиль GitHub](https://github.com/SofiaMakashova/ITMO_VKR)]
Научный руководитель: Воскресенский А.В.


