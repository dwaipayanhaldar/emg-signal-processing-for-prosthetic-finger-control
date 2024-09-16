import numpy as np
import pandas as pd
import math
from statsmodels.regression.linear_model import yule_walker


def features_estimation(signal, frame, step):
    """
    Compute time, frequency and time-frequency features from signal.
    :param signal: numpy array signal.
    :param channel_name: string variable with the EMG channel name in analysis.
    :param frame: sliding window siz e
    :param step: sliding window step size
    :param plot: bolean variable to plot estimated features.

    :return: total_feature_matrix -- python Dataframe with .
    :return: features_names -- python list with

    """

    signal = np.array(signal)

    if len(signal.shape)==1:
        signal = signal[np.newaxis,:]

    datasize = signal.shape[1]
    Nsignals = signal.shape[0]
    numwin = int(np.floor((datasize - frame) / step) + 1)

    NFPC = 16

    feat = np.zeros((numwin, Nsignals * NFPC))

    for i in range(Nsignals):
        td_matrix = time_features_estimation(signal[i,:], frame, step)
        ar_matrix = autoregressive_coefficients(signal[i,:], frame, step)
        hjorth_matrix = hjort_parameters(signal[i,:], frame, step)
        total_matrix = np.column_stack((td_matrix, ar_matrix, hjorth_matrix))
        feat[:,i*NFPC:(i+1)*NFPC] = total_matrix


    return feat


def time_features_estimation(signal, frame, step):
    """
    Compute time features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size.
    :param step: sliding window step size.

    :return: time_features_matrix: narray matrix with the time features stacked by columns.
    """
    iemg = []
    mav = []
    mav1 = []
    mav2 = []
    ssi = []
    variance = []
    rms = []
    wl = []
    dasdv = []

    for i in range(frame, signal.size+1, step):
        x = signal[i - frame:i]

        iemg.append(np.sum(abs(x)))  # Integral

        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value

        mav1.append(mav1_func(x)) # Mean Absolute Value Type 1

        mav2.append(mav2_func(x)) # Mean Absolute Value Type 2

        ssi.append(np.sum(x**2)) # Simple Square Integral

        variance.append(np.var(x, ddof = 1))

        rms.append(np.sqrt(np.mean(x ** 2)))

        wl.append(np.sum(abs(np.diff(x))))  # Wavelength

        dasdv.append(
            math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value

    time_features_matrix = np.column_stack((iemg, mav, mav1, mav2, ssi, variance, rms, wl, dasdv))
    return time_features_matrix


def autoregressive_coefficients(signal, frame, step):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """
    ar1 = []
    ar2 = []
    ar3 = []
    ar4 = []

    for i in range(frame, signal.size+1, step):
        x = signal[i - frame:i]

        rho, sigma = yule_walker(x, order=4)

        ar1.append(rho[0])
        ar2.append(rho[1])
        ar3.append(rho[2])
        ar4.append(rho[3])

    autoregressive_coefficients_matrix = np.column_stack((ar1, ar2, ar3, ar4))

    return autoregressive_coefficients_matrix


def hjort_parameters(signal, frame, step):
    """
    Compute time-frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: h_wavelet: list
    """
    hjorth_1 = []
    hjorth_2 = []
    hjorth_3 = []

    for i in range(frame, signal.size+1, step):
        x = signal[i - frame:i]
        hjorth_1.append(np.var(x))
        h2, h3 = hjorth(x)
        hjorth_2.append(h2)
        hjorth_3.append(h3)

    hjort_parameter_matrix = np.column_stack((hjorth_1, hjorth_2, hjorth_3))

    return hjort_parameter_matrix

def mav1_func(x):
    mav1_sum = 0
    for i in range(x.size):
        if int((x.size*0.25)-1) <= i <= int((x.size*0.75)-1):
            w_i = 1
        else:
            w_i = 0.5
        mav1_sum += w_i*np.absolute(x[i])

    return mav1_sum/x.size

def mav2_func(x):
    mav2_sum = 0
    for i in range(x.size):
        if int((x.size*0.25)-1) <= i <= int((x.size*0.75)-1):
            w_i = 1
        elif i < int(x.size*0.25 -1):
            w_i = (4*(i+1))/x.size
        else:
            w_i = (4*(i+1-x.size))/x.size
        mav2_sum += w_i*np.absolute(x[i])

    return mav2_sum/x.size


def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity
