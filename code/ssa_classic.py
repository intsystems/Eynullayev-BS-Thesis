from scipy.linalg import svd
from scipy.linalg import norm
from scipy.linalg import inv
import numpy as np
import scipy.linalg as linalg

from ssa_base import SSA_Base

class SSA_classic(SSA_Base):
    """class for making SSA decomposition and prediction of given 1-dimensional time series
    """
    def __init__(self, signal: list, L: int):
        """
        :param list signal: initial time series
        :param int L: size of the sliding-window
        """
        super().__init__(L)
        # save given time series
        self.t_s = signal

        # singular values and factors of SVD. Not supposed to be change explicitly
        self.weights: np.ndarray = None
        self._left_factors: np.ndarray = None
        self._right_factors: np.ndarray = None

        # grouping of factors variable
        self.grouping = []

        # array of predicted values
        self._forecast = []

    
    def set_factors_grouping(self, grouping: list):
        """set grouping of factors for futher decomposition

        :param list grouping: list of lists of indices. Each list - its own group
        """
        self.grouping = grouping


    def decompose_tm(self):
        """making svd and save factors and singular values
        """
        # construct trajectory matrix
        traj_matrix = self._build_traj_matrix(self.t_s, self.L)

        # applying svd. It already returns singular values in decsent order!
        U, s_v, V_tr = svd(traj_matrix, full_matrices=False)

        # saving singular values in descend order
        self.weights = s_v
        # saving factor vectors
        self._left_factors = U
        self._right_factors = V_tr


    def dispose_factors(self, indices: tuple):
        """remove factors with given indices

        Args:
            indices (tuple): factors to dispose     

        Returns:
            relative residual new and old traj. matrices
        """
        # computing relative residual between new and old traj. matrices
        rel_residual = np.sqrt(np.sum(self.weights[indices] ** 2) / np.sum(self.weights ** 2))

        self.weights = np.delete(self.weights, indices)
        self._left_factors = np.delete(self._left_factors, indices, axis=1)
        self._right_factors = np.delete(self._right_factors, indices, axis=0)

        return rel_residual


    def decompose_signal(self) -> tuple:
        """ extract signals-components according to grouping set

        :raises ValueError: if grouping is not set
        :return tuple: list of signal-components, list of hankelization errors - absolute and reletive
        """
        if self.grouping == []:
            raise ValueError('Grouping is not set.')

        # hankelization residuals array
        hankel_resids_abs = np.empty(len(self.grouping))
        hankel_resids_rel = np.empty(len(self.grouping))

        # signal-components storage
        component_signals = []

        # constructing component-signals
        for group_ind, group in enumerate(self.grouping):
            # constructing trajectory matrix
            cur_traj_matr = np.zeros(shape=(self._left_factors.transpose()[0].shape[0], self._right_factors[0].shape[0]),
                                      dtype=np.float32)

            # summing skeletones
            for ind in group:
                cur_traj_matr += self.weights[ind] * np.outer(self._left_factors.transpose()[ind], self._right_factors[ind])

            # hankelizing traj. matrix
            temp = cur_traj_matr.copy()
            self._hankelize_matrix(cur_traj_matr)

            # safe residual
            hankel_resids_abs[group_ind] = norm(temp - cur_traj_matr, ord='fro')
            hankel_resids_rel[group_ind] = hankel_resids_abs[group_ind] / norm(temp, ord='fro')

            # extract 1d signal
            component_signals.append(self._extract_ts_from_tm(cur_traj_matr))

        return component_signals, hankel_resids_abs, hankel_resids_rel
    
    
    def get_prediction(self):
        """method to get all available predictions
        """
        return self._forecast.copy()
    

    def remove_last_prediction(self, k: int = None):
        """remove last k predictions

        :param int k: number of prediction to delete
        :raises ValueError: if trying delete more then available
        """
        if k is None:
            self._forecast = []
            return
        if k > len(self._forecast):
            raise ValueError('Deleting more values then available.')
        
        self._forecast = self._forecast[:-k]
    

    def predict_next(self, k: int) -> list:
        """predict next k values of time series considering already existing forecasts. It's not always possible.

        :param int k: prediction horizon
        """
        if k > self.L - len(self.weights):
            raise ValueError('Impossible to predict more then (L-r) values')
        
        # get known part of U matrix
        U_known = self._left_factors[:self.L - k]
        # get pred part of U matrix
        U_pred = self._left_factors[self.L - k:]

        # build known part of x
        x_known = np.empty(self.L - k)
        # number of values to take from given time series
        from_init = np.maximum(0, (self.L - k) - len(self._forecast))
        # number of values to take from already predicted values
        from_pred = (self.L - k) - from_init
        # fill x_known
        if from_pred == 0:
            x_known = self.t_s[-from_init:]
        elif from_init == 0:
            x_known = self._forecast[-from_pred:]
        else:
            x_known[:from_init] = self.t_s[-from_init:]
            x_known[from_init:] = self._forecast

        try:
            temp = inv(np.eye(k, k) - U_pred @ U_pred.T)
        except linalg.LinAlgError:
            raise linalg.LinAlgError('Impossible to get prediction due to singularity. Try decrease k.')

        # get prediction
        x_pred = temp @ U_pred @ U_known.T @ x_known
        # save prediction
        self._forecast.extend(x_pred.tolist())

        return x_pred