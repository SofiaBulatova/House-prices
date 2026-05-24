from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from Config import config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from NN import NeuralNet

# Создаем модель
def get_model(model_name, params, config=None):
    if model_name == 'stacking':
        estimator_names = params['estimators']
        estimators = []
        for name in estimator_names:
            base_params = dict(config.models[name])
            base_model = get_model(name, base_params, config)
            estimators.append((name, base_model))
        final_name = params['final_estimator']
        final_params = dict(config.models[final_name])
        final_estimator = get_model(final_name, final_params, config)
        cv = params.get('cv', 5)
        return StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=cv)
    elif model_name == "linear_regression":
        return LinearRegression(**params)
    elif model_name == "lasso":
        return Lasso(**params)
    elif model_name == "ridge":
        return Ridge(**params)
    elif model_name == "elasticnet":
        return ElasticNet(**params)
    elif model_name == "knn":
        return KNeighborsRegressor(**params)
    elif model_name == "decision_tree":
        return DecisionTreeRegressor(**params)
    elif model_name == "random_forest":
        return RandomForestRegressor(**params)
    elif model_name == "catboost":
        return CatBoostRegressor(**params)
    elif model_name == "lightgbm":
        return LGBMRegressor(**params)
    elif model_name == "xgboost":
        return XGBRegressor(**params)
    elif model_name == "neural_network":
        return NeuralNet(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Создаем экземпляр модели
def get_models_dict(config):
    models = {}
    for name, params in config.models.items():
        p = dict(params)
        model = get_model(name, p, config=config)
        # Маштабируем признаки
        if name in ['linear_regression', 'lasso', 'ridge', 'elasticnet', 'knn']:
            model = make_pipeline(StandardScaler(), model)
        elif name == 'stacking':
            model = make_pipeline(StandardScaler(), model)
        models[name] = model
    return models
