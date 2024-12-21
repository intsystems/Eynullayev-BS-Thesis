import numpy as np
import scipy.linalg as linalg
import tensorly as tl

from ssa_base import SSA_Base
from ssa_classic import SSA_classic


class t_SSA(SSA_Base):
    """ class for tSSA method for decomposition and prediction of multidimensional time series
    """
    def __init__(self, L: int, signals: list, cpd_rank: int):
        super().__init__(L)

        self.t_s_list = signals

        # Matrices of tensor CP decomposition with meaning as in article. Not supposed to be change explicitly
        self.weights: np.ndarray = None
        self._left_factors: np.ndarray = None
        self._right_factors: np.ndarray = None

        # list for each signal containing its available factor nums 
        self._signals_available_factors = [np.arange(cpd_rank) for i in range(len(signals))]

        # list of grouping of factors for each signal
        self.grouping = [[] for i in range(len(signals))]

        # array of predicted values for each signal
        self._forecast = [[] for i in range(len(signals))]

        # supportive vectors for quick predictions
        self._pred_vecs = [[] for i in range(len(signals))]

        # contains error of CP decomposition algorithm
        self.cpd_err_abs = None
        self.cpd_err_rel = None

        # desirable rank of CP decomposition. Can only be set in __init__() and decompose_tt()
        self._cpd_rank = cpd_rank


    def get_cpd_rank(self):
        return self._cpd_rank

    
    def set_factors_grouping(self, grouping: list, signal_num: int):
        """set grouping of factors for particular signal. Numeration must consider disposed factors if exist.

        :param list grouping: list of list of indicies corresponding to number of factors to group
        :param int signal_num: number of signal to set grouping for
        """
        self.grouping[signal_num] = grouping

    
    def dispose_factors(self, indices: list, signal_num: int):
        """remove factors with given indices for particular signal. Numeration must be as after CPD. 
            Does not actually delete factors!

        :param list indices: factors to remove
        :return: relative residual between new and old traj. matrices 
        """
        self._signals_available_factors[signal_num] = \
                                        np.delete(self._signals_available_factors[signal_num], indices)

        # computing relative residual between new and old traj. matrices
        rel_residual = linalg.norm(self.weights[signal_num][indices], ord=2) / linalg.norm(self.weights[signal_num], ord=2)

        # reset supportive vectors to recalculate it futher
        self._pred_vecs[signal_num] = []

        return rel_residual

    
    def decompose_tt(self, cpd_rank: int=None, random_state: int = None):
        """build and perform CPD of trajectory tensor. Safe all factors and decomposition error

        :param int cpd_rank: desirable rank of CPD, default to None <=> get rank from object init()
        :param random_state int: parameter for CPD-algorithm. Warrant deterministic behaviour
        """
        # construct traj. tensor
        traj_matr = self._construct_traj_tensor()
        traj_matr_copy = tl.copy(traj_matr)

        # set to default value if not set manually
        if cpd_rank is None:
            cpd_rank = self._cpd_rank
        else:
            self._cpd_rank = cpd_rank

        # perform CPD and safe factors
        cpd_decomp, reconstr_err = tl.decomposition.parafac(traj_matr, rank=cpd_rank, normalize_factors=True, init='random', \
                                              return_errors=True, verbose=0, random_state=random_state)

        factor_norms = cpd_decomp[0]
        
        self.weights = tl.copy(cpd_decomp[1][2])

        self._left_factors = tl.copy(cpd_decomp[1][0])
        self._right_factors = tl.copy(cpd_decomp[1][1].T)
        # put factor norms into c-vectors
        self.weights *= factor_norms

        # save reconstruction error 
        self.cpd_err_rel = reconstr_err[-1]
        self.cpd_err_abs = tl.norm(traj_matr_copy, order=2) * reconstr_err[-1]

        # debug
        #print(f'Relative residual norm = {self.cpd_err_rel}')
        #print(f'Absolute residual norm = {self.cpd_err_abs}')


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

        for i in range(len(self.t_s_list)):
            cur_sig_len = len(self.t_s_list[i])

            # set factors and grouping
            ssa_classic_obj.weights, ssa_classic_obj._left_factors, ssa_classic_obj._right_factors = \
                                                                                self.get_available_factors(i)

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


    def predict_next(self):
        """get prediction for every signal for one step futher

        :return _type_: prediction array and residuals array (from least squares)
        """
        pred_arr = []

        for i in range(len(self.t_s_list)):
            cur_pred = self._predict_next_sigwise(i)

            pred_arr.append(cur_pred)
            self._forecast[i].append(cur_pred)

        return pred_arr

    
    def _predict_next_sigwise(self, sig_num: int):
        """get prediction for one step futher using least squares
        """
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

        # compute supportive vector if not yet
        if len(self._pred_vecs[sig_num]) == 0:
            self._compute_pred_vec(sig_num)

        # get prediction
        pred = self._pred_vecs[sig_num] @ x_known

        # return prediction and residuals of lambda solution
        return pred
    

    def _compute_pred_vec(self, sig_num: int):
        """method to compute vector needed for tSSA prediction

        :raises linalg.LinAlgError: if unable to reverse A_kn.T @ A_kn
        """
        # get available factors
        _, avail_l_factors, _ = self.get_available_factors(sig_num)
        
        # compute needed vector
        A_kn = avail_l_factors[:-1]
        A_pred = avail_l_factors[-1]
        try:
            pred_vec = A_pred @ linalg.inv(A_kn.T @ A_kn) @ A_kn.T
        except linalg.LinAlgError as er:
            raise linalg.LinAlgError('Cannot compute vector-string for prediction')
        
        self._pred_vecs[sig_num] = pred_vec


    def get_available_factors(self, sig_num: int) -> tuple:
        """return weights and factors for given signal considering disposed ones
        """
        # obtain avialaible indices 
        avail_indxs = self._signals_available_factors[sig_num]

        # return available (not deleted) factors 
        return self.weights[sig_num][avail_indxs], self._left_factors[:, avail_indxs], self._right_factors[avail_indxs, :]
        

    def _construct_traj_tensor(self):
        # initializing traj. tensor
        K = len(self.t_s_list[0]) - self.L + 1
        traj_tens = tl.zeros(shape=(self.L, K, len(self.t_s_list)))

        # filling it with traj. matrices
        for i, t_s in enumerate(self.t_s_list):
            cur_traj_matrix = self._build_traj_matrix(t_s, self.L)
            traj_tens[0:, 0:, i] = cur_traj_matrix

        return traj_tens 

    def non_sequential_pred(self, xs_known):
        # Эта функция позволяет выполнить прогноз с помощью базиса в собственном пространстве
        # для произвольных векторов задержки набора временных рядов длины L - 1

        preds = []

        for sig_num, x_known in enumerate(xs_known):
            # compute supportive vector if not yet
            if len(self._pred_vecs[sig_num]) == 0:
                self._compute_pred_vec(sig_num)
            
            pred = self._pred_vecs[sig_num] @ x_known
            preds.append(pred)

        return preds