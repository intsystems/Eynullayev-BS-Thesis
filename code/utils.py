import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from tSSA import t_SSA
from sklearn.model_selection import train_test_split

def tangent_space_cov(ts, w_size, k=1, est='scm'):
    cov = Covariances(estimator=est)
    X = np.array([ts.to_numpy().T[:, i * k:i * k + w_size] for i in range(((ts.shape[0] - w_size + 1) // k + 1) - 1)])
    est_cov = cov.transform(X)
    tangent_space = TangentSpace()
    tsp_X = tangent_space.fit_transform(est_cov)

    return tsp_X

def optimal_cpd_rank(train_data, val_data, w_size, random_state=42):
    cpd_ranks = np.arange(5, 30 + 1, 5)
    forecast_results = {key: {"mse": None, "mape": None} for key in cpd_ranks}

    # parameter for detereministic behaviour of tSSA
    random_state = 42

    for cpd_rank in cpd_ranks:
        print(f"CP rank = {cpd_rank}")
        t_ssa_obj = t_SSA(w_size, train_data.T, cpd_rank)

        # make svd for common matrix, extract factors and singular values
        t_ssa_obj.decompose_tt(random_state=random_state)
        print(f"CPD-error = {t_ssa_obj.cpd_err_rel}")

        t_ssa_obj.remove_last_predictions()

        # get prediction for cuurent number of factors left
        forecast_tssa = np.empty(val_data.shape)

        for i in range(val_data.shape[0]):
            forecast_tssa[i] = np.array(t_ssa_obj.predict_next())

        # get MSE for every signal
        signals_mse_tssa = np.mean((forecast_tssa - val_data) ** 2, axis=0)
        # get MAPE for every signal
        signals_mape_tssa = np.mean(np.abs((forecast_tssa - val_data) / val_data), axis=0)

        forecast_results[cpd_rank]["mse"] = signals_mse_tssa
        forecast_results[cpd_rank]["mape"] = signals_mape_tssa

        print(f'MSE: {signals_mse_tssa}; Mean by signals = {np.mean(signals_mse_tssa):e}')
        print(f'MAPE: {signals_mape_tssa}; Mean by signals = {np.mean(signals_mape_tssa):e}')

    return forecast_results
    
def get_tssa_train_fc(t_ssa, train_ts, w_size, k=1):
    delay_vectors = np.array([train_ts.T[:, i * k:i * k + (w_size - 1)] for i in range(((train_ts.shape[0] - (w_size - 1) + 1) // k + 1) - 1)])
    delay_vectors_1 = delay_vectors[1:]
    predicted_ts = []

    for delay in delay_vectors_1:
        preds = t_ssa.non_sequential_pred(delay)
        predicted_ts.append(preds)

    predicted_ts = np.array(predicted_ts[:-1]) # убираем последний, т.к. предсказание выходит за рамки обучающей выборки
    actual_train_ts = train_ts[w_size:, :]

    return predicted_ts, actual_train_ts

def tssa_cov_split(predicted_ts, actual_ts, tsp_X):
    tsp_X_1 = tsp_X[:-1]
    X = np.concatenate([predicted_ts, tsp_X_1], axis=1)
    X_train, X_test, y_train,  y_test = train_test_split(X, actual_ts, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test

