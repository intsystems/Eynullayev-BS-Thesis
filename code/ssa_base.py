"""
    introducing basic class for ssa, containing common complemetery methods
"""
from abc import abstractmethod
import numpy as np

class SSA_Base:
    """ Base class for SSA methods in the package
    """

    def __init__(self, L: int) -> None:
        # basic parameter for SSA
        self.L = L

 
    @staticmethod
    def _build_traj_matrix(ts: np.ndarray, L: int) -> np.ndarray:
        """building trajectory matrix from 1d time series

        :param np.ndarray ts: 1d signal
        :param int L: size of delay vector
        :return np.ndarray: trajectory matrix
        """
        # number of windowsize samples
        K = len(ts) - L + 1

        # making trajectory matrix
        traj_matrix = np.empty((L, K), float)
        for i in range(K):
            traj_matrix.transpose()[i] = ts[i:i + L]

        return traj_matrix


    @staticmethod
    def _extract_ts_from_tm(ar: np.ndarray):
        """method extract time series out of trajectory matrix

        :param np.ndarray ar: trajectory_matrix
        :return _type_: 1d time series
        """
        first_part = ar.transpose()[0][:-1]
        second_part = ar[-1]
        return np.hstack((first_part, second_part))


    @staticmethod
    def _hankelize_matrix(ar: np.ndarray):
        """method to hankelize single matrix

        :param np.ndarray ar: matrix to hankelize
        """
        # above the main antidiagonal and on it
        for i in range(ar.shape[0]):
            cur_sum = 0
            for j in range(min(i + 1, ar.shape[1])):
                cur_sum += ar[i - j][j]
            avg = cur_sum / (i + 1)

            for j in range(min(i + 1, ar.shape[1])):
                ar[i - j][j] = avg

        # below the main antidiagonal
        for i in range(1, ar.shape[1]):
            cur_sum = 0
            j = 0
            while i + j != ar.shape[1] and ar.shape[0] - j != 0:
                cur_sum += ar[(ar.shape[0] - 1) - j][i + j]
                j += 1
            avg = cur_sum / j

            j = 0
            while i + j != ar.shape[1] and ar.shape[0] - j != 0:
                ar[(ar.shape[0] - 1) - j][i + j] = avg
                j += 1