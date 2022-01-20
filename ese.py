import numpy as np
from scipy.stats import genpareto
from pot import pot


def ese(w, gev_window, filter_len, pot_method):
    """
    :param w: input weights of adaptive filter
    :param gev_window: number of samples that are proceeded via P-O-T method
    :param filter_len: length of adaptive filter (obsolete)
    :param pot_method: identifier of P-O-T method (1,2,3,4,5)
    :return: values of Extreme Seeking Entropy
    """
    dw = np.copy(w)
    dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
    dw_count = int(dw.shape[0])

    hpp = np.ones((dw_count - gev_window, filter_len))
    for i in range(gev_window, dw.shape[0]):
        if i % 100 == 0:
            pass  # print((str(datetime.now())), " processing: ", i)
        for j in range(filter_len):
            poted_values = pot(dw[i - gev_window:i, j], pot_method)
            if dw[i, j] > poted_values[-1]:
                fit = genpareto.fit(poted_values, floc=[poted_values[-1]])
                if dw[i, j] >= fit[1]:
                    hpp[i - gev_window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-20

    ese_value = -np.log10(np.prod(hpp, axis=1))
    return ese_value
