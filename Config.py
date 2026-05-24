from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 42,
    },
    'paths': {
        'train_path': 'Data/train_processed.csv',
        'test_path': 'Data/test_processed.csv',
        'output': 'output'
    },
    'models': {
        'lasso': {'alpha': 0.1, 'max_iter': 20000},
        'ridge': {'alpha': 0.1, 'max_iter': 20000},
        'elasticnet': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 20000},
        'knn': {'n_neighbors': 5},
        'decision_tree': {'max_depth': 10, 'random_state': '${general.seed}'},
        'random_forest': {'n_estimators': 100, 'random_state': '${general.seed}'},
        'catboost': {'iterations': 2000, 'verbose': False, 'random_state': '${general.seed}', 'learning_rate': 0.03,
            'depth': 6, 'loss_function': 'RMSE'},
        'lightgbm': {'n_estimators': 2000, 'random_state': '${general.seed}', 'learning_rate': 0.03, 'num_leaves': 31},
        'xgboost': {'n_estimators': 2000, 'verbosity': 0, 'random_state': '${general.seed}'},
        'stacking': {'type': 'stacking', 'final_estimator': 'lasso', 'cv': 5, 'estimators': ['catboost', 'lightgbm']},
        'neural_network': {
            'hidden_dims': [128, 62],
            'dropout': 0.5,
            'lr': 0.001,
            'epochs': 200,
            'batch_size': 32,
            'patience': 15,
            'val_fraction': 0.15,
            'random_state': '${general.seed}',
            'step_size': 10,
            'gamma': 0.5
        },
    },
    'split': {
        'methods':['cv', 'holdout'],
        'cv':{'n_splits': 5, 'shuffle': True, 'random_state': '${general.seed}'},
        'holdout':{'test_size': 0.2, 'random_state': '${general.seed}'}

    },
    'id_col': 'Id',
    'target_col': 'SalePrice'

}
config = OmegaConf.create(config)