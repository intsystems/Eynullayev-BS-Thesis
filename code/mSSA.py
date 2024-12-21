from scipy.linalg import svd
from scipy.linalg import norm
from scipy.linalg import inv
import numpy as np
import scipy.linalg as linalg

from ssa_base import SSA_Base
from ssa_classic import SSA_classic


class m_SSA(SSA_Base):
    """ class of mSSA method for decomposition and prediction of multidimensional time series
    """
    def __init__(self, L: int, signals: list):
        """
        :param list signals: initial time series
        :param int L: size of the sliding-window
        """
        super().__init__(L)

        self.t_s_list = signals

        # singular values and factors of SVD (for common matrix). Not supposed to be change explicitly
        self.weights: np.ndarray = None
        self._left_factors: np.ndarray = None
        self._right_factors: np.ndarray = None

        # list of grouping of factors for each signal
        self.grouping = [[] for i in range(len(signals))]

        # array of predicted values for each signal
        self._forecast = [[] for i in range(len(signals))]

        # supportive vector for quick predictions
        self._pred_vec = []

    
    def set_factors_grouping(self, grouping: list, signal_num: int):
        """set grouping of factors for particular signal. Numeration must consider disposed factors if exist.

        :param list grouping: list of list of indecies corresponding to number of factors to group
        :param int signal_num: number of signal to set grouping for
        """
        self.grouping[signal_num] = grouping

    
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

        # reset supportive vector
        self._pred_vec = []

        return rel_residual

    
    def decompose_tt(self):
        """making svd and save factors and singular values of common
             trajectory matrix (plays a role of trajectory tensor here).
        """
        traj_matrix_list = []
        # construct trajectory matrix for individual signals
        for t_s in self.t_s_list:
            traj_matrix_list.append(self._build_traj_matrix(t_s, self.L))

        # construct common trajectory matrix
        common_traj_matrix = np.concatenate(traj_matrix_list, axis=1)
        # applying svd. It already returns singular values in decsent order!
        U, s_v, V_tr = svd(common_traj_matrix, full_matrices=False)

        # saving singular values in descend order
        self.weights = s_v
        # saving factor vectors
        self._left_factors = U
        self._right_factors = V_tr


    def decompose_signals(self) -> tuple:
        """extract signals-components according to grouping set for each signal

        :raises ValueError: if grouping for some signal is not set
        :return tuple: lists of signal-components, lists of hankelization errors - absolute and reletive
        """
        # output variables
        hankel_resids_abs = [[] for i in range(len(self.t_s_list))]
        hankel_resids_rel = [[] for i in range(len(self.t_s_list))]
        component_signals = [[] for i in range(len(self.t_s_list))]

        # constuct classic_ssa object ans use it
        ssa_classic_obj = SSA_classic([], self.L)
        ssa_classic_obj._left_factors = self._left_factors
        ssa_classic_obj.weights = self.weights

        temp = 0

        for i in range(len(self.t_s_list)):
            cur_sig_len = len(self.t_s_list[i])

            # set factors and grouping
            ssa_classic_obj._right_factors = self._right_factors[:, temp:temp + (cur_sig_len - self.L + 1)]
            temp += cur_sig_len - self.L + 1
            try:
                ssa_classic_obj.set_factors_grouping(self.grouping[i])
            except ValueError:
                raise ValueError(f'Grouping for signal {i + 1} is not set.')
            
            # decompose
            cur_comp_sigs, cur_hankel_resids_abs, cur_hankel_resids_rel = ssa_classic_obj.decompose_signal()
            # save results
            component_signals[i] = cur_comp_sigs
            hankel_resids_abs[i] = cur_hankel_resids_abs
            hankel_resids_rel[i] = cur_hankel_resids_rel

        return component_signals, hankel_resids_abs, hankel_resids_rel
    

    def get_predictions(self):
        """method to get all available predictions for every signal
        """
        return self._forecast.copy()
    

    def remove_last_predictions(self, k: int = None):
        """remove last k predictions for all signals

        :param int k: number of prediction to delete
        :raises ValueError: if trying delete more then available
        """
        if k is None:
            self._forecast = [[] for i in range(len(self.t_s_list))]
            return
        if k > len(self._forecast[0]):
            raise ValueError('Deleting more values then available.')
        
        self._forecast = [self._forecast[i][:k] for i in range(len(self.t_s_list))]


    def predict_next(self) -> list:
        """get prediction for one step futher for every signal

        :return list: list of predictions
        """
        prediction_list = []
        
        # compute supportive vec if not yet
        if len(self._pred_vec) == 0:
            U_kn = self._left_factors[:-1]
            U_pr = self._left_factors[-1]

            try:
                pred_vec = linalg.inv(np.eye(1, 1) - U_pr @ U_pr.T)[0] * U_pr @ U_kn.T
            except linalg.LinAlgError as er:
                raise linalg.LinAlgError('Cannot compute vector-string for prediction')

            self._pred_vec = pred_vec

        for sig_num in range(len(self.t_s_list)):
            # build known part of x
            x_known = np.empty(self.L - 1)
            # number of values to take from given time series
            from_init = np.maximum(0, (self.L - 1) - len(self._forecast[sig_num]))
            # number of values to take from already predicted values
            from_pred = (self.L - 1) - from_init
            # fill x_known
            if from_pred == 0:
                x_known = self.t_s_list[sig_num][-from_init:]
            elif from_init == 0:
                x_known = self._forecast[sig_num][-from_pred:]
            else:
                x_known[:from_init] = self.t_s_list[sig_num][-from_init:]
                x_known[from_init:] = self._forecast[sig_num]

            cur_pred = (self._pred_vec @ x_known)
            prediction_list.append(cur_pred)
            self._forecast[sig_num].append(cur_pred)

        return prediction_list