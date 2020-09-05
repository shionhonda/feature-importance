import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from tqdm import tqdm

np.random.seed(0)

def f_rand(X):
    return np.random.randn(len(X))

def f_noised_mean(X):
    return (np.sum(X, axis=1) + np.random.randn(len(X))) / ((X.shape[1]+1)**0.5)

def gen_data_ctrl(f, n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    y = f(X)
    return X, y

def gen_data_card(f, n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    y = f(X)
    for i in range(1, n_features):
        X[:,i] = X[:,i].round(n_features-1-i)
    return X, y

def gen_data_dup(f, n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    X[:,-1] = X[:,-2]
    y = f(X)
    return X, y

def gen_data_colinear(f, n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    X[:,-1] = (X[:,-2] + np.random.normal(0, 0.5, len(X))) / (1.25**0.5)
    y = f(X)
    return X, y

def gini_importance(gen_data, f, n_iters, n_features, n_train=1000):
    arr_imp = np.zeros((n_iters, n_features))
    for i in tqdm(range(n_iters)):
        X_train, y_train = gen_data(f, n_train, n_features)
        model = LGBMRegressor(importance_type="gain")
        model.fit(X_train, y_train)
        arr_imp[i] = model.feature_importances_
    return arr_imp

def split_importance(gen_data, f, n_iters, n_features, n_train=1000):
    arr_imp = np.zeros((n_iters, n_features))
    for i in tqdm(range(n_iters)):
        X_train, y_train = gen_data(f, n_train, n_features)
        model = LGBMRegressor(importance_type="split")
        model.fit(X_train, y_train)
        arr_imp[i] = model.feature_importances_
    return arr_imp

def drop_importance(gen_data, f, n_iters, n_features, n_train=1000, n_val=200):
    arr_imp = np.zeros((n_iters, n_features))
    for i in tqdm(range(n_iters)):
        X_train, y_train = gen_data(f, n_train, n_features)
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        X_val, y_val = gen_data(f, n_val, n_features)
        y_pred = model.predict(X_val)
        baseline = mean_squared_error(y_val, y_pred)
        for j in range(n_features):
            model = LGBMRegressor()
            model.fit(np.delete(X_train, j, axis=1), y_train)
            y_pred = model.predict(np.delete(X_val, j, axis=1))
            drop = mean_squared_error(y_val, y_pred)
            arr_imp[i, j] = drop - baseline
    return arr_imp

def perm_importance(gen_data, f, n_iters, n_features, n_train=1000, n_val=200):
    arr_imp = np.zeros((n_iters, n_features))
    for i in tqdm(range(n_iters)):
        X_train, y_train = gen_data(f, n_train, n_features)
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        X_val, y_val = gen_data(f, n_val, n_features)
        arr_imp[i] = permutation_importance(model, X_val, y_val)['importances_mean']
    return arr_imp