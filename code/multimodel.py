import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from journal_style import EnableJournalStylePlotting
import scipy

class LocalModel():

    def __init__(self, k: int, L: int):
        self.k = k
        self.L = L

    @staticmethod
    def HankelMatrix(X, L):
        N = X.shape[0]
        return scipy.linalg.hankel(X[ : N - L + 1], X[N - L : N])

    def fit(self, ts):
        self.n = ts.shape[1]
        self.T = ts.shape[0]
        self.ts = ts

        self.hankel_matrices = {i: self.HankelMatrix(self.ts[:, i], self.L) for i in range(self.n)}

        return self

    def predict(self, h):
        preds = {i: [] for i in range(self.n)}

        for i in range(self.n):
            for _ in range(h):
                    distances = np.sum((self.hankel_matrices[i] - self.hankel_matrices[i][-1]) ** 2, axis=1)
                    nn = np.argsort(distances)
                    nn = nn[nn != len(self.hankel_matrices[i]) - 1]
                    knn = nn[:self.k]
                    nn_targets = self.hankel_matrices[i][knn + 1, -1]
                    preds[i].append(np.mean(nn_targets))
                    self.hankel_matrices[i] = np.append(self.hankel_matrices[i], [list(self.hankel_matrices[i][-1, 1:]) + [preds[i][-1]]], axis=0)
        
        self.preds = np.concatenate([np.array(preds[i]).reshape((len(preds[i]), 1)) for i in range(self.n)], axis=1)

        return self.preds
    
    def mae(self, ts_test):
        mae_local = {}
        for i in range(self.n):
            mae_local[i] = np.mean(np.abs((self.preds.T[i] - ts_test[:, i])), axis=0)

            print(f'MAE: {mae_local[i]}')

        print(f'Mean by signals: MAE = {np.mean(list(mae_local.values()))}')

        return mae_local

    def mse(self, ts_test, start, end):
        test_mse = np.mean((self.preds[start:end, :] - ts_test[start:end, :]) ** 2, axis=0) # mse на тестовой выборке
        return test_mse
    
    def mape(self, ts_test):
        mape_local = {}
        for i in range(self.n):
            mape_local[i] = np.mean((np.abs(self.preds[i].T - ts_test[:, i])) / np.abs(ts_test[:, i]), axis=0)

            print(f'MSE: {mape_local[i]}')

        print(f'Mean by signals: MAPE = {np.mean(list(mape_local.values()))}')

        return mape_local
    
    def plot_pred(self, ts_test):
        for i in range(self.n):
            with EnableJournalStylePlotting():
                fig, ax = plt.subplots(figsize=(15, 8))

                ax.plot(ts_test.T[i], label='test')
                ax.plot(np.array(self.preds.T[i]).T, label='predict')

                ax.grid(True)
                ax.legend()

                ax.tick_params(axis='x', labelsize=32)
                ax.tick_params(axis='y', labelsize=32)
                if i == 0:
                    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
                    ax.set_title('Local model prediction')
                
        plt.show()



            

class LocalMultiModel():

    def __init__(self, L: int, f_model: object, F_model: object, cov_est: str):
        self.L = L
        self.f_model = f_model
        # self.g_model = g_model
        self.F_model = F_model
        self.cov_est = cov_est
        self.cov = Covariances(estimator=self.cov_est)
        self.tangent_space = TangentSpace()


    def fit(self, ts_train, h, cov_use=True, eps=0.001):
        '''
            Функция принимает на вход временной ряд и параметр h. На выходе возвращается обученная модель F. 
            Метод последовательно выполняет следующее:

            1) Вычисляет матрицы ковариации и переводит их в касательное пространство.
            2) Делит временной ряд на две части history, train. Параметр h задает длину train. 
               ts_history служит для выполнения прогноза на train с помощью f_model. На train будет обучаться мультимодель F_model.
            3) Таким образом прогноз f_model на train -- один из входов для модели F_model. Эту часть датасета создает функция _train_data_from _f. 
               Второй вход -- tangent space вектор в предыдущий момент времени. Эту часть датасета создает функция _train_data_from_g.
        '''
        self.cov_use = cov_use # флаг использования ковариаций в прогнозе
        self.history = ts_train[:ts_train.shape[0] - h, :] # создаем history
        self.train = ts_train[ts_train.shape[0] - h:, :] # создаем train
        self.h = h

        self.n = self.history.shape[1] # количество временных рядов
        self.T = self.history.shape[0] # длина истории

        if self.cov_use:
            X = np.array([ts_train.T[:, i:i + self.L] for i in range(((ts_train.shape[0] - self.L + 1) + 1) - 1)]) # массив срезов точек фазовых траекторий
            self.cov_ts = self.cov.transform(X) + np.identity(self.n) * eps # строим матрицы ковариации
            self.tgt_cov_ts = self.tangent_space.fit_transform(self.cov_ts) # переводим их в касательное пространство

        self._train_data_from_f() # f_preds
        if self.cov_use:
            self._train_data_from_g() # g_preds
        self._F_train() # обучаем F на полученных данных

        return self # возвращаем обученную на данных модель

    def _train_data_from_f(self):
        '''
            Эта функция делает прогноз на train части датасета по history.
        '''
        self.f_model.fit(self.history) # обучаемся на истории
        self.f_preds = self.f_model.predict(self.h) # предсказываем train

        return self.f_preds

    def _train_data_from_g(self, eps=0.001):
        '''
            Эта функция считает tangent space векторы для прогноза f_model на train. 
            Альтернатива: (брать известные tg_space векторы, посчитанные на реальных значениях).
        '''
        
        if self.cov_use == True:

            data_ts = np.concatenate([self.history[self.T - self.L:, :], self.f_preds], axis=0)
            X_g = np.array([data_ts.T[:, i:i + self.L] for i in range(((data_ts.shape[0] - self.L + 1) + 1) - 1)][:-1]) # массив срезов точек фазовых траекторий
            cov_ts = self.cov.transform(X_g) + np.identity(self.n) * eps # строим матрицы ковариации
            self.g_preds = self.tangent_space.fit_transform(cov_ts) # переводим их в касательное пространство

            print(f'g: {self.g_preds.shape[0]}, f: {self.f_preds.shape[0]}')

            assert self.g_preds.shape[0] == self.f_preds.shape[0]
        
        else:

            self.g_preds = []

        

    def _F_train(self):
        '''
            Эта функция обучает мультимодель F_model по f_preds, g_preds на 90% данных из train. 
            Также она выводит качество обученной модели на обучающей выборке и тестовой.
        '''

        F_y = self.train # выход F

        if self.cov_use:

            F_X = np.concatenate([self.f_preds, self.g_preds], axis=1) # вход F

        else:

            F_X = self.f_preds

        X_train, X_test, self.y_train,  self.y_test = train_test_split(F_X, F_y, test_size=0.1, random_state=42, shuffle=False) # сплиттим датасет в отношении 9:1

        self.scaler = StandardScaler() 
        self.scaler.fit(X_train)
        X_train_transformed = self.scaler.transform(X_train) # Отшкалировали обучающую выборку
        X_test_transformed = self.scaler.transform(X_test) # Отшкалировали тестовую выборку

        self.F_model = self.F_model.fit(X_train_transformed, self.y_train) # Обучаемся

        self.y_pred_train = self.F_model.predict(X_train_transformed) # Предсказываем на обучающей выборке
        self.y_pred_test = self.F_model.predict(X_test_transformed) # Предсказываем на тестовой выборке

    def F_quality(self):
        '''
            Эта функция позволяет сравнить улучшение точности прогноза при использовании F_model.
            Она выводит точность обученной F_model на обучающей выборке и на тестовой. 
            Также точность f_model на обучающей выборке и тесте. 

        '''
        output = {'train_mse': [],
                  'test_mse': [],
                  'train_mean': [],
                  'test_mean': [], 
                  'local_train_mse': [],
                  'local_test_mse': [],
                  'local_train_mean': [],
                  'local_test_mean': [],}
        
        train_mse = np.mean((self.y_pred_train - self.train[:self.y_pred_train.shape[0]]) ** 2, axis=0) # mse на обучающей выборке
        test_mse = np.mean((self.y_pred_test - self.train[self.y_pred_train.shape[0]:]) ** 2, axis=0) # mse на тестовой выборке
        local_train_mse = self.f_model.mse(self.train, 0, self.y_pred_train.shape[0])
        local_test_mse = self.f_model.mse(self.train, self.y_pred_train.shape[0], self.train.shape[0])

        output['train_mse'].append(train_mse) 
        output['train_mean'].append(np.mean(train_mse))
        output['test_mse'].append(test_mse) 
        output['test_mean'].append(np.mean(test_mse))

        output['local_train_mse'].append(local_train_mse) 
        output['local_train_mean'].append(np.mean(local_train_mse))
        output['local_test_mse'].append(local_test_mse)
        output['local_test_mean'].append(np.mean(local_test_mse))

        return output

    def plot_fit_pred(self):
        '''
            Эта функция рисует графики прогнозов на train с помощью F_model и простой f_model.
        '''
        preds = np.concatenate([self.y_pred_train, self.y_pred_test], axis=0)
        for i in range(self.n):
            with EnableJournalStylePlotting():
                fig, ax = plt.subplots(figsize=(15, 8))

                ax.plot(self.train.T[i], label='test')
                ax.plot(preds.T[i], label='predict')

                ax.grid(True)
                ax.legend()

                ax.tick_params(axis='x', labelsize=32)
                ax.tick_params(axis='y', labelsize=32)
                if i == 0:
                    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
                    ax.set_title('F model prediction')
        plt.show()        
                

        self.f_model.fit(self.history)
        self.f_model.predict(self.h)
        self.f_model.plot_pred(self.train)
        

    def predict(self, pred_h, eps=0.001):
        '''
            Эта функция выполняет прогноз на горизонт pred_h с помощью обученной мультимодели.
        '''
        ts = np.concatenate([self.history, self.train], axis=0)
        ts_size = ts.shape[0]
        f_preds = self.f_model.fit(ts).predict(pred_h)

        data_ts = np.concatenate([ts[ts_size - self.L:, :], f_preds], axis=0)
        if self.cov_use:
            X_g = np.array([data_ts.T[:, i:i + self.L] for i in range(((data_ts.shape[0] - self.L + 1) + 1) - 1)][:-1]) # массив срезов точек фазовых траекторий
            cov_ts = self.cov.transform(X_g) + np.identity(self.n) * eps # строим матрицы ковариации
            g_preds = self.tangent_space.fit_transform(cov_ts) # переводим их в касательное пространство

            assert g_preds.shape[0] == f_preds.shape[0]

            X_test = np.concatenate([f_preds, g_preds], axis=1)
        else:
            X_test = f_preds

        X_test_transformed = self.scaler.transform(X_test)
        F_preds = self.F_model.predict(X_test_transformed)
        
        return F_preds
