import numpy as np
from tslearn.metrics import dtw_path


def dtw_distance(s, t):
    """
    :param s:
    :param t:
    :return distm:
    """

    ns, nt = len(s), len(t)
    distm = np.ones(shape=(ns + 1, nt + 1)) * np.inf
    distm[0, 0] = 0

    for i in range(0, ns):
        for j in range(0, nt):
            cost = np.linalg.norm(s[i] - t[j])
            distm[i + 1, j + 1] = cost + min([distm[i, j + 1], distm[i + 1, j], distm[i, j]])
    return distm


def tdi_tdm_2(s, t, w=None, inner=None):
    """

    :param s: vector of the reference series
    :param t: vector of the test series
    :param w: (optional)  size of the sliding window in the inner product (only needed if inner=1)
    :param inner: (optional) set to 1 to use the inner product instead of the Euclidean distance
    :return:
    """

    distm = dtw_distance(s, t)
    i, j = len(s), len(t)
    i_l, j_l = [], []

    while i > 0 and j > 0:
        neighbors = [distm[i - 1, j], distm[i, j - 1], distm[i - 1, j - 1]]
        val_i = np.array([i - 1, i, i - 1])
        val_j = np.array([j, j - 1, j - 1])
        a = np.where(neighbors == np.min(neighbors))[0]
        b = np.argmin(np.abs(val_i[a] - val_j[a]))  # In case of equality, take the first
        c = a[b]
        i = val_i[c]
        j = val_j[c]
        i_l.append(i)
        j_l.append(j)

    i_l = np.flip(np.array(i_l)) + 1
    j_l = np.flip(np.array(j_l)) + 1

    tdi = 2 / (max(j_l) * max(i_l)) * np.trapz(abs(j_l - i_l), j_l)

    indices = np.where(j_l >= i_l)[0]
    tdi_late = abs(np.trapz(j_l[indices], j_l[indices] - i_l[indices]) / np.trapz(j_l, abs(j_l - i_l)))
    tdi_late = np.nan_to_num(tdi_late, nan=0.5)
    tdm = 2 * (tdi_late - 0.5)
    return tdi, tdm


def tdi_tdm(s, t):
    """

    :param s:
    :param t:
    :return:
    """

    optimal_path = dtw_path(s, t)[0]
    i_l = np.array([i for i, _ in optimal_path]) + 1
    j_l = np.array([j for _, j in optimal_path]) + 1

    tdi = 2 / (max(j_l) * max(i_l)) * np.trapz(abs(j_l - i_l), j_l)

    indices = np.where(j_l >= i_l)[0]
    tdi_late = abs(np.trapz(j_l[indices], j_l[indices] - i_l[indices]) / np.trapz(j_l, abs(j_l - i_l)))
    tdi_late = np.nan_to_num(tdi_late, nan=0.5)
    tdm = 2 * (tdi_late - 0.5)
    return tdi, tdm


def squared_error(test, forecast):
    """"""
    difference = np.subtract(test, forecast)
    error = np.square(difference)
    return error


def absolute_error(test, forecast):
    """"""
    difference = np.subtract(test, forecast)
    error = np.abs(difference)
    return error


def test_tdi_tdm():
    s = np.array([0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0])
    t = np.array([0, 0, 0, 0, 0, 0, 15, 15, 15, 0, 0, 0, 0, 0, 25, 25, 25, 25, 25, 25, 25, 0, 0, 0, 0, 0, 25, 25, 25, 25, 25, 0, 0, 0, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0])

    print("Matlab version:")
    tdi, tdm = tdi_tdm_2(s, t)
    print("TDI      :", tdi)
    print("TDM      :", tdm)

    print("\ntslearn optimal path version:")
    tdi, tdm = tdi_tdm_2(s, t)
    print("TDI      :", tdi)
    print("TDM      :", tdm)


if __name__ == '__main__':
    test_tdi_tdm()
