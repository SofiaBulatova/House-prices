import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from Config import config

# Выбор метода и формирование Датафрейма
def evaluate_all_models(models_dict, X, y, config):
    rows = []
    for method in config.split.methods:
        if method == 'cv':
            cv_config = config.split.cv
            for name, model in models_dict.items():
                rmse = get_kfold(model, X, y, n_folds=cv_config.n_splits,
                                shuffle=cv_config.shuffle,
                                random_state=cv_config.random_state)
                rows.append({'model': name, 'method': method, 'rmse': rmse})
        elif method == 'holdout':
            holdout_config = config.split.holdout
            for name, model in models_dict.items():
                rmse = holdout(model, X, y, test_size=holdout_config.test_size,
                                       random_state=holdout_config.random_state)
                rows.append({'model': name, 'method': method, 'rmse': rmse})
    df = pd.DataFrame(rows)
    df = df.sort_values('rmse', ascending=True).reset_index(drop=True)
    return df

# Осуществляем KFold
def get_kfold(model, X, y, n_folds, shuffle, random_state):
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    return rmse_scores.mean()

# Осуществляем train_test_split
def holdout(model, X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))