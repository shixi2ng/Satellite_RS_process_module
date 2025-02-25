import os
import numpy as np
import datetime
import copy
from scipy.stats import chi2
from scipy.signal import band_stop_obj
from sklearn.linear_model import Lasso


global w
w = 2 * np.pi / 365.25


def autoTSFit(x, y, df):
    """
    Auto Trends and Seasonal Fit between breaks
    Args:
    - x: Julian day (e.g., [1, 2, 3])
    - y: Predicted reflectances (e.g., [0.1, 0.2, 0.3])
    - df: Degree of freedom (num_c)

    Returns:
    - fit_coeff: Fitted coefficients
    - rmse: Root mean square error
    - v_dif: Differences between observed and predicted values
    """

    # Initialize fit coefficients
    fit_coeff = np.zeros(8)

    # Build X matrix
    X = np.zeros((len(x), df - 1))
    X[:, 0] = x

    if df >= 4:
        X[:, 1] = np.cos(w * x)
        X[:, 2] = np.sin(w * x)

    if df >= 6:
        X[:, 3] = np.cos(2 * w * x)
        X[:, 4] = np.sin(2 * w * x)

    if df >= 8:
        X[:, 5] = np.cos(3 * w * x)
        X[:, 6] = np.sin(3 * w * x)

    # Lasso fit with lambda = 20
    lasso = Lasso(alpha=20, fit_intercept=True)
    lasso.fit(X, y)

    # Set fit coefficients
    fit_coeff[0] = lasso.intercept_
    fit_coeff[1: df] = lasso.coef_

    # Predicted values
    yhat = autoTSPred(x, fit_coeff)
    fit_diff = y - yhat
    fit_rmse = np.linalg.norm(fit_diff) / np.sqrt(len(x) - df)

    return fit_coeff, fit_rmse, fit_diff


def autoTSPred(outfitx, fit_coeff):
    """
    Auto Trends and Seasonal Predict
    Args:
    - outfitx: Julian day (e.g., [1, 2, 3])
    - fit_coeff: Fitted coefficients

    Returns:
    - outfity: Predicted reflectances
    """
      # annual cycle

    # Construct the design matrix
    X = np.column_stack([
        np.ones_like(outfitx),  # overall ref
        outfitx,                # trending
        np.cos(w * outfitx),    # seasonality
        np.sin(w * outfitx),
        np.cos(2 * w * outfitx),  # bimodal seasonality
        np.sin(2 * w * outfitx),
        np.cos(3 * w * outfitx),  # trimodal seasonality
        np.sin(3 * w * outfitx)
    ])

    outfity = X @ fit_coeff  # matrix multiplication

    return outfity


def update_cft(i_span, n_times, min_num_c, mid_num_c, max_num_c, num_c):
    """
    Determine the time series model based on the span of observations.

    Args:
    - i_span: Span of observations
    - n_times: Multiplier for the number of coefficients
    - min_num_c: Minimum number of coefficients
    - mid_num_c: Middle number of coefficients
    - max_num_c: Maximum number of coefficients
    - num_c: Current number of coefficients

    Returns:
    - update_num_c: Updated number of coefficients
    """
    if i_span < mid_num_c * n_times:
        # Start with 4 coefficients model
        update_num_c = min(min_num_c, num_c)
    elif i_span < max_num_c * n_times:
        # Start with 6 coefficients model
        update_num_c = min(mid_num_c, num_c)
    else:
        # Start with 8 coefficients model
        update_num_c = min(max_num_c, num_c)

    return update_num_c


def TrendSeasonalFit_v12_30Line(doy_arr, inform_arr,  num_c: int = 8, Thr_change_detect=0.99, min_obs4stable: int=6):

    # Define constants %
    # maximum number of coefficient required
    # 2 for tri-modal; 2 for bi-modal; 2 for seasonality; 2 for linear;
    min_num_c = 4 # Minimum number of coefficients
    mid_num_c = 6 # Middle number of coefficients
    max_num_c = 8 # Maximum number of coefficients
    n_times = 3 # Multiplier for the number of coefficients
    number_fitcurve = 0
    num_yrs = 365.25
    num_byte = 2
    yrs_thr = 1

    # Result list
    CCDC_result = []

    # Define var
    num_time, num_band = inform_arr.shape[0], inform_arr.shape[1]
    x_arr = doy_arr
    y_arr = inform_arr
    deltay_arr = y_arr[1:] - y_arr[:-1]
    adj_rmse = np.median(np.abs(deltay_arr), axis=0)
    i_end = n_times * min_num_c
    i_start = 1

    # Fitcurve indicator
    BL_train = 0 # No break found at the beggning
    number_fitcurve += 1
    record_fitcurve = number_fitcurve
    Thrmax_change_detect = chi2.ppf(1 - 1e-6, num_band)

    # CCDC procedure
    while i_end <= len(x_arr) - min_obs4stable:
        i_span = i_end - i_start + 1
        time_span = (x_arr[i_end] - x_arr[i_start]) / num_yrs
        if i_span >= n_times * min_num_c and time_span >= yrs_thr:

            # initializing model
            if BL_train == 0:

                fit_coeff = np.zeros((max_num_c, num_band))
                bands_fit_rmse = np.zeros(num_band)
                v_dif = np.zeros(num_band)
                bands_fit_diff = np.zeros((i_end - i_start + 1, num_band))

                # fitting
                for band_index in range(num_band):
                    fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index] = autoTSFit(x_arr[i_start:i_end + 1], y_arr[i_start:i_end + 1, band_index], min_num_c)

                    # normalized to z - score
                    # minimum rmse
                    mini_rmse = max(adj_rmse[band_index], bands_fit_rmse[band_index])
                    # compare the first clear obs
                    v_start = bands_fit_diff[0, band_index] / mini_rmse
                    # compare the last clear observation
                    v_end = bands_fit_diff[-1, band_index] / mini_rmse
                    # normalized slope values
                    v_slope = fit_coeff[1, band_index] * (x_arr[i_end] - x_arr[i_start]) / mini_rmse
                    # difference in model initialization
                    v_dif[band_index] = abs(v_slope) + abs(v_start) + abs(v_end)

                v_dif = np.linalg.norm(v_dif) ** 2

                # find stable start for each curve
                if v_dif > Thr_change_detect:
                    # MOVE FORWARD
                    i_start += 1
                    i_end += 1

                else:
                    # model ready! Count difference of i for each itr
                    BL_train = 1
                    i_count = 0

                    # find the previous break point
                    if number_fitcurve == record_fitcurve:
                        i_break = 1
                    else:
                        i_break = np.where(x_arr >= CCDC_result[number_fitcurve - 1]['t_break'])[0][0]

                    if i_start > i_break:
                        # model fit at the beginning of the time series
                        for i_ini in range(i_start - 1, i_break - 1, -1):

                            if i_start - i_break < min_obs4stable:
                                ini_conse = i_start - i_break
                            else:
                                ini_conse = min_obs4stable

                            # change vector magnitude
                            v_dif = np.zeros((ini_conse, num_band))
                            v_dif_mag = copy.deepcopy(v_dif)
                            vec_mag = np.zeros(ini_conse)

                            for i_conse in range(ini_conse):
                                for band_index in range(num_band):
                                    v_dif_mag[i_conse, band_index] = y_arr[i_ini - i_conse, band_index] - autoTSPred(x_arr[i_ini - i_conse], fit_coeff[:, band_index])
                                    mini_rmse = max(adj_rmse[band_index], bands_fit_rmse[band_index])
                                    v_dif[i_conse, band_index] = v_dif_mag[i_conse, band_index] / mini_rmse
                                vec_mag[i_conse] = np.linalg.norm(v_dif[i_conse, :]) ** 2

                            if min(vec_mag) > Thr_change_detect:
                                break
                            elif vec_mag[0] > Thrmax_change_detect:
                                x_arr = np.delete(x_arr, i_ini)
                                y_arr = np.delete(y_arr, i_ini, axis=0)
                                i_end -= 1

                            # update new_i_start if i_ini is not a confirmed break
                            i_start = i_ini

                        if number_fitcurve == record_fitcurve and i_start - i_break >= min_obs4stable:
                            fit_coeff = np.zeros((max_num_c, num_band))
                            bands_fit_rmse = np.zeros(num_band)
                            qa = 10

                            for band_index in range(num_band):
                                fit_coeff[:, band_index], bands_fit_rmse[band_index] = autoTSFit(x_arr[i_break: i_start], y_arr[i_break: i_start, band_index], min_num_c)

                            CCDC_result[number_fitcurve - 1].update({
                                't_end': x_arr[i_start - 1],
                                'pos': num_time - 1,
                                'coeffs': fit_coeff,
                                'bands_fit_rmse': bands_fit_rmse,
                                't_break': x_arr[i_start],
                                'change_prob': 1,
                                't_start': x_arr[0],
                                'category': qa + min_num_c,
                                'num_obs': i_start - i_break,
                                'magnitude': -np.median(v_dif_mag, axis=0)
                            })
                            number_fitcurve += 1

            # continuous monitoring started!!!
            if BL_train == 1:

                IDs = np.arange(i_start, i_end + 1)
                i_span = i_end - i_start + 1

                # determine the time series model
                update_num_c = update_cft(i_span, n_times, min_num_c, mid_num_c, max_num_c, num_c)

                # initial model fit when there are not many obs
                if i_count == 0 or i_span <= max_num_c * n_times:

                    # update i_count at each interation
                    i_count = x_arr[i_end] - x_arr[i_start]

                    # defining computed variables
                    fit_coeff = np.zeros((max_num_c, num_band))
                    bands_fit_rmse = np.zeros(num_band)
                    bands_fit_diff = np.zeros((len(IDs), num_band))
                    qa = 0

                    for band_index in range(num_band):
                        fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index] = autoTSFit(x_arr[IDs], y_arr[IDs, band_index], update_num_c)

                    CCDC_result[number_fitcurve - 1].update({
                        't_start': x_arr[i_start],
                        't_end': x_arr[i_end],
                        't_break': 0,
                        'pos': num_time - 1,
                        'coeffs': fit_coeff,
                        'bands_fit_rmse': bands_fit_rmse,
                        'change_prob': 0,
                        'num_obs': i_end - i_start + 1,
                        'category': qa + update_num_c,
                        'magnitude': np.zeros(num_band)
                    })

                    v_dif = np.zeros((min_obs4stable, num_band))
                    v_dif_mag = v_dif.copy()
                    vec_mag = np.zeros(min_obs4stable)

                    for i_conse in range(min_obs4stable):
                        for band_index in range(num_band):
                            v_dif_mag[i_conse, band_index] = y_arr[i_end + i_conse, band_index] - autoTSPred(x_arr[i_end + i_conse], fit_coeff[:, band_index])
                            mini_rmse = max(adj_rmse[band_index], bands_fit_rmse[band_index])
                            v_dif[i_conse, band_index] = v_dif_mag[i_conse, band_index] / mini_rmse
                        vec_mag[i_conse] = np.linalg.norm(v_dif[i_conse, :]) ** 2
                    IDsOld = IDs.copy()

                else:

                    if x_arr[i_end] - x_arr[i_start] >= 1.33 * i_count:
                        i_count = x_arr[i_end] - x_arr[i_start]
                        fit_coeff = np.zeros((max_num_c, num_band))
                        bands_fit_rmse = np.zeros(num_band)
                        bands_fit_diff = np.zeros((len(IDs), num_band))
                        qa = 0

                        for band_index in range(num_band):
                            fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index] = autoTSFit(x_arr[IDs], y_arr[IDs, band_index], update_num_c)

                        CCDC_result[number_fitcurve - 1].update({
                            'coeffs': fit_coeff,
                            'bands_fit_rmse': bands_fit_rmse,
                            'num_obs': i_end - i_start + 1,
                            'category': qa + update_num_c
                        })
                        IDsOld = IDs.copy()

                    # record time of curve end
                    CCDC_result[number_fitcurve - 1]['t_end'] = x_arr[i_end]
                    num4rmse = int(n_times * CCDC_result[number_fitcurve - 1].category)
                    temp_change_rmse = np.zeros(num_band)
                    d_rt = x_arr[IDsOld] - x_arr[i_end + min_obs4stable]
                    d_yr = np.abs(np.round(d_rt / num_yrs) * num_yrs - d_rt)
                    sorted_index = np.argsort(d_yr)[: num4rmse]

                    for band_index in range(num_band):
                        temp_change_rmse[band_index] = np.linalg.norm(bands_fit_diff[IDsOld[sorted_index] - IDsOld[0] + 1, band_index]) / np.sqrt(num4rmse - CCDC_result[number_fitcurve - 1].category)

                    v_dif[:-1, :] = v_dif[1:, :]
                    v_dif[-1, :] = 0
                    v_dif_mag[:-1, :] = v_dif_mag[1:, :]
                    v_dif_mag[-1, :] = 0
                    vec_mag[:-1] = vec_mag[1:]
                    vec_mag[-1] = 0

                    for band_index in range(num_band):
                        v_dif_mag[-1, band_index] = y_arr[i_end + min_obs4stable, band_index] - autoTSPred(x_arr[i_end + min_obs4stable], fit_coeff[:, band_index])
                        mini_rmse = max(adj_rmse[band_index], temp_change_rmse[band_index])
                        v_dif[-1, band_index] = v_dif_mag[-1, band_index] / mini_rmse
                    vec_mag[-1] = np.linalg.norm(v_dif[-1, :]) ** 2

                if min(vec_mag) > Thr_change_detect:
                    CCDC_result[number_fitcurve - 1]['t_break'] = x_arr[i_end + 1]
                    CCDC_result[number_fitcurve - 1]['change_prob'] = 1
                    CCDC_result[number_fitcurve - 1]['magnitude'] = np.median(v_dif_mag, axis=0)

                    number_fitcurve += 1
                    i_start = i_end + 1
                    BL_train = 0

                elif vec_mag[0] > Thrmax_change_detect:
                    x_arr = np.delete(x_arr, i_end + 1)
                    y_arr = np.delete(y_arr, i_end + 1, axis=0)
                    i_end -= 1

    # Two ways for processing the end of the time series
    if BL_train == 1:
        #  if no break find at the end of the time series
        #  define probability of change based on conse

        for i_conse in range(min_obs4stable - 1, -1, -1):
            if vec_mag[i_conse] <= Thr_change_detect:
                id_last = i_conse
                break

        CCDC_result[number_fitcurve - 1]['change_prob'] = (min_obs4stable - id_last) / min_obs4stable
        CCDC_result[number_fitcurve - 1]['t_end'] = x_arr[-min_obs4stable + id_last]

        if min_obs4stable > id_last:
            CCDC_result[number_fitcurve - 1]['t_break'] = x_arr[-min_obs4stable + id_last + 1]
            CCDC_result[number_fitcurve - 1]['magnitude'] = np.median(v_dif_mag[id_last + 1:min_obs4stable, :], axis=0)

    elif BL_train == 0:

        #  2) if break find close to the end of the time series
        #  Use [min_obs4stable,min_num_c*n_times+min_obs4stable) to fit curve

        if number_fitcurve == record_fitcurve:
            i_start = 1
        else:
            i_start = np.where(x_arr >= CCDC_result[number_fitcurve - 1]['t_break'])[0][0]

        # Check if there is enough data
        if len(x_arr[i_start:]) >= min_obs4stable:
            # Define computed variables
            fit_cft = np.zeros((max_num_c, num_band))
            rmse = np.zeros(num_band)
            qa = 20

            for band_index in range(num_band):
                fit_cft[:, band_index], rmse[band_index] = autoTSFit(x_arr[i_start:], y_arr[i_start:, band_index], min_num_c)

            # Record information
            CCDC_result.append({
                't_start': x_arr[i_start],
                't_end': x_arr[-1],
                't_break': 0,
                'pos': num_time - 1,
                'coefs': fit_cft,
                'rmse': rmse,
                'change_prob': 0,
                'num_obs': len(x_arr[i_start:]),
                'category': qa + min_num_c,
                'magnitude': np.zeros(num_band)
            })


