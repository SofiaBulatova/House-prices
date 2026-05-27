# House Prices
ML-пайплайн для задачи Kaggle House Prices.
## Структура проекта
* Data/train_processed.csv - train после EDA.
* Data/test_processed.csv - test после EDA.
* output/best_model_info.json - информация о лучшей модели, включая ее гиперпараметры.
* output/catboost.joblib - лучшая модель с сохраненными параметрами и гиперпараметрами.
* output/validation_resalts.csv - accuracy всех моделей.
* output/submission.csv - файл для загрузки на Kaggle.
* Config.py - ключевые настройки.
* Data analysis.py - анализ данных.
* Feature engineering.py - очистка данных, выбросы.
* NN.py - нейронная сеть.
* Prediction.py - предсказание лучшей модели.
* Split.py - K-Fold, train_test_split.
* main.py - основной файл для запуска.
* models.py - все модели
* train.py - обучение моделей, выбор лучшей и ее сохранение.
## Использованные модели
В пайплайне использовались следующие модели:
* Линейная регрессия с Lasso.
* Линейная регрессия с Ridge.
* Линейная регрессия с ElasticNet.
* KNN.
* Дерево решений.
* Случайный лес.
* CatBoost.
* lightgbm.
* xgboost.
* Stacking c CatBoost и lightgbm.
* Нейронная сеть.
## Результаты
По результатам валидации была выбрана CatBoost с методом holdout и RMSE 20273.66.  
На Kaggle Public Score - 0,12412
## Установка и запуск
1. **Клонировать репозиторий**
   ```bash
   git clone https://github.com/SofiaBulatova/House-prices.git
   cd House-prices
2. **Создать и активировать окружение**
   ```bash
   python -m venv venv
   venv\Scripts\activate
4. **Установить зависимости**
   ```bash
   pip install -r requirements.txt
5. **Запустить обучение**
   ```bash
   python main.py
