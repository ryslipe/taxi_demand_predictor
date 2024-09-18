from src.model import get_pipeline

hyperparams = {
    "metric": 'mae',
    "verbose": -1,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "min_child_samples": 20
}

pipeline = get_pipeline(**hyperparams)
print(