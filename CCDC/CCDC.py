import traceback
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import chi2
from sklearn.linear_model import Lasso
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# CCDC 12.30 version - Zhe Zhu, EROS USGS
# It is based on 7 bands fitting for Iterative Seasonal, Linear, and Break Models
# This function works for analyzing one line of time series pixel

global w
w = 2 * np.pi / 365.25


def glmnet_lasso_single(X, y, lambda_val=1.0, fit_intercept=True, standardize=True, max_iter=10000, tol=1e-4):
    """
    Simplified glmnet-style Lasso for a single lambda value.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
    - y: ndarray, shape (n_samples,)
    - lambda_val: float, lambda (alpha) value for Lasso
    - fit_intercept: bool, whether to fit and recover intercept
    - standardize: bool, whether to standardize X
    - max_iter: int, maximum iterations
    - tol: float, tolerance for convergence

    Returns:
    - dict with keys: 'lambda', 'beta', 'a0', 'df', 'rss', 'r2'
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).flatten()

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    model = Lasso(alpha=lambda_val, fit_intercept=fit_intercept,
                  max_iter=max_iter, tol=tol)
    model.fit(X_scaled, y)

    beta = model.coef_
    intercept = model.intercept_ if fit_intercept else 0.0

    y_pred = model.predict(X_scaled)
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - rss / tss if tss > 1e-12 else 0.0

    return {
        'lambda': lambda_val,
        'beta': beta,         # shape (n_features,)
        'a0': intercept,      # scalar
        'df': np.sum(np.abs(beta) > 1e-8),
        'rss': rss,
        'r2': r2
    }

# def lassofit(x, y, optimum_lambda):
#
#     nobs, nvars = x.shape
#
#     # Set default options.
#     options = {
#         'weights': [],
#         'alpha': 1.0,
#         'lambda': optimum_lambda,  # optimum lambda for CCDC
#         'standardize': True,
#         'thresh': 1E-4,
#         'HessianExact': False,
#     }
#
#     # ======= 参数准备 =======
#     weights = options.get('weights', np.ones(nobs))
#     alpha = options.get('alpha', 1.0)
#     lambda_seq = options.get('lambda', None)
#     standardize = options.get('standardize', True)
#     thresh = options.get('thresh', 1e-4)
#
#     if len(options['weights']) == 0:
#         options['weights'] = np.ones((nobs, 1))
#
#     # ======= 中心化 + 权重下的 null deviance =======
#     ybar = np.dot(y.T, weights) / np.sum(weights)
#     nulldev = np.dot((y - ybar)**2, weights) / np.sum(weights)
#
#     # ======= 是否标准化输入 =======
#     if standardize:
#         x_mean = np.average(x, axis=0, weights=weights)
#         x_std = np.std(x, axis=0)
#         x_std[x_std == 0] = 1.0
#         x = (x - x_mean) / x_std
#
#     # convert lambda to ascending order like glmnet.m does internally
#     lambda_list = np.sort(lambda_list)[::-1]
#     nlam = len(lambda_list)
#
#     # storage
#     a0 = np.zeros(nlam)  # intercepts
#     beta = np.zeros((nvars, nlam))  # coefficients
#     dev = np.zeros(nlam)  # R²
#     df = np.zeros(nlam, dtype=int)  # degrees of freedom (nonzero coef)
#
#     # for each lambda value
#     for i in range(nlam):
#         lam = lambda_list[i]
#         model = Lasso(alpha=lam, fit_intercept=True, tol=thresh, max_iter=10000)
#         model.fit(x, y)
#
#         a0[i] = model.intercept_
#         beta[:, i] = model.coef_
#         df[i] = np.sum(model.coef_ != 0)
#
#         y_pred = model.predict(x)
#         rss = np.dot((y - y_pred) ** 2, weights)
#         dev[i] = 1 - rss / np.dot((y - ybar) ** 2, weights)
#
#     # return in a glmnet-style dictionary
#     fit = {
#         'a0': a0,  # intercepts
#         'beta': beta,  # coefficients matrix (nvars x nlam)
#         'df': df,  # non-zero coefficient count
#         'dev': dev,  # R^2 per lambda
#         'nulldev': nulldev,  # total sum of squares
#         'lambda': lambda_list,  # regularization values
#         'dim': (nvars, nlam),  # dimensions of beta
#         'class': 'elnet'  # consistent with glmnet
#     }
#
#     return fit

# def lass2(x, y, optimum_lambda):
#     nobs, nvars = x.shape
#
#     # Set default options.
#     options = {
#         'weights': [],
#         'alpha': 1.0,
#         'lambda': optimum_lambda,  # optimum lambda for CCDC
#         'standardize': True,
#         'thresh': 1E-4,
#         'HessianExact': False,
#     }
#
#     if len(options['weights']) == 0:
#         options['weights'] = np.ones((nobs, 1))
#     ybar = np.dot(y.T, options['weights']) / np.sum(options['weights'])
#     nulldev = np.dot((y - ybar) ** 2, options['weights']) / np.sum(options['weights'])
#
#
#     isd = float(options['standardize'])
#     thresh = options['thresh']
#     lambda_ = options['lambda']
#     flmin = 1.0
#     ulam = -np.sort(-lambda_)
#     nlam = len(lambda_)
#     parm = options['alpha']
#
#     # ======= 权重处理：拟合时 sqrt(w) 调整 ====
#     x_weighted = x * np.sqrt(weights[:, np.newaxis])
#     y_weighted = y * np.sqrt(weights)
#
#     # ======= 模型拟合（路径方式）=======
#     if lambda_seq is None:
#         alphas, coefs, _ = lasso_path(x_weighted, y_weighted, eps=1e-3, fit_intercept=True)
#         intercepts = np.mean(y_weighted) - np.dot(np.mean(x_weighted, axis=0), coefs)
#         rsq = 1 - np.sum((y_weighted[:, None] - x_weighted @ coefs) ** 2, axis=0) / np.var(y_weighted) / len(y)
#     else:
#         alphas = np.sort(lambda_seq)[::-1]
#         coefs = []
#         intercepts = []
#         rsq = []
#         for a in alphas:
#             model = Lasso(alpha=a, fit_intercept=True, max_iter=10000, tol=thresh)
#             model.fit(x_weighted, y_weighted)
#             coefs.append(model.coef_)
#             intercepts.append(model.intercept_)
#             rsq.append(model.score(x_weighted, y_weighted))
#         coefs = np.array(coefs).T
#         intercepts = np.array(intercepts)
#         rsq = np.array(rsq)
#
#     # ======= 构造结果结构体（仿 MATLAB）=======
#     fit = {
#         'a0': intercepts,
#         'beta': coefs,
#         'dev': rsq,
#         'nulldev': nulldev,
#         'df': np.sum(np.abs(coefs) > 1e-8, axis=0),
#         'lambda': alphas,
#         'npasses': 0,
#         'jerr': 0,
#         'dim': coefs.shape,
#         'class': 'elnet'
#     }
#
#     return fit
#
#
# def fix_lam(lam):
#     new_lam = lam.copy()
#     llam = np.log(lam)
#     new_lam[0] = np.exp(2 * llam[1] - llam[2])
#     return new_lam



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
    lasso = Lasso(alpha=20/10000, fit_intercept=True, max_iter=25000)
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


def TrendSeasonalFit_v12_30Line(doy_arr, inform_arr, num_c: int = 8, Thr_change_detect=0.99, min_obs4stable: int=6, ):

    if doy_arr.shape[0] != inform_arr.shape[0]:
        raise Exception('The doy and inform is not consistent in size！')

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
    yrs_thr = 4

    # Result list
    CCDC_result = []

    # Define var
    num_time, num_band = inform_arr.shape[0], inform_arr.shape[1]
    x_arr = doy_arr
    y_arr = inform_arr
    deltay_arr = y_arr[1:] - y_arr[:-1]
    adj_rmse = np.median(np.abs(deltay_arr), axis=0)
    i_end = n_times * min_num_c
    i_start = 0

    # Fitcurve indicator
    # record the start of the model initialization (0=>initial;1=>done)
    BL_train = 0 # No break found at the beggning
    number_fitcurve += 1
    record_fitcurve = copy.deepcopy(number_fitcurve)
    Thr_change_detect = chi2.ppf(Thr_change_detect, num_band)
    Thrmax_change_detect = chi2.ppf(1 - 0.000001, num_band)

    try:
        # CCDC procedure
        while i_end < len(x_arr) - min_obs4stable:
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
                        continue

                    else:
                        # model ready! Count difference of i for each itr
                        BL_train = 1
                        i_count = 0

                        # find the previous break point
                        if number_fitcurve == record_fitcurve:
                            i_break = 0
                        else:
                            i_break = np.where(x_arr >= CCDC_result[number_fitcurve - 2]['t_break'])[0][0]

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
                                        v_dif_mag[i_conse, band_index] = y_arr[i_ini - i_conse, band_index] - autoTSPred(x_arr[i_ini - i_conse], fit_coeff[:, band_index])[0]
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
                            bands_fit_diff = np.zeros((i_start - i_break, num_band))
                            qa = 10

                            for band_index in range(num_band):
                                fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index]  = autoTSFit(x_arr[i_break: i_start], y_arr[i_break: i_start, band_index], min_num_c)

                            if number_fitcurve - 2 > len(CCDC_result):
                                CCDC_result[number_fitcurve - 2].update({
                                    't_start': x_arr[i_start],
                                    't_end': x_arr[i_end],
                                    't_break': 0,
                                    'pos': num_time - 1,
                                    'coeffs': fit_coeff,
                                    'bands_fit_rmse': bands_fit_rmse,
                                    'change_prob': 0,
                                    'num_obs': i_end - i_start + 1,
                                    'category': qa + min_num_c,
                                    'magnitude': np.zeros(num_band)
                                })
                            else:
                                CCDC_result.append({
                                    't_start': x_arr[i_start],
                                    't_end': x_arr[i_end],
                                    't_break': 0,
                                    'pos': num_time - 1,
                                    'coeffs': fit_coeff,
                                    'bands_fit_rmse': bands_fit_rmse,
                                    'change_prob': 0,
                                    'num_obs': i_end - i_start + 1,
                                    'category': qa + min_num_c,
                                    'magnitude': np.zeros(num_band)
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

                        if number_fitcurve - 2 > len(CCDC_result):
                            CCDC_result[number_fitcurve - 2].update({
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
                        else:
                            CCDC_result.append({
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
                                v_dif_mag[i_conse, band_index] = y_arr[i_end + i_conse, band_index] - autoTSPred(x_arr[i_end + i_conse], fit_coeff[:, band_index])[0]
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

                            CCDC_result[number_fitcurve - 2].update({
                                'coeffs': fit_coeff,
                                'bands_fit_rmse': bands_fit_rmse,
                                'num_obs': i_end - i_start + 1,
                                'category': qa + update_num_c
                            })
                            IDsOld = IDs.copy()

                        # record time of curve end
                        CCDC_result[number_fitcurve - 2]['t_end'] = x_arr[i_end]
                        num4rmse = int(n_times * CCDC_result[number_fitcurve - 2]['category'])
                        temp_change_rmse = np.zeros(num_band)
                        d_rt = x_arr[IDsOld] - x_arr[i_end + min_obs4stable]
                        d_yr = np.abs(np.round(d_rt / num_yrs) * num_yrs - d_rt)
                        sorted_index = np.argsort(d_yr)[: num4rmse]

                        for band_index in range(num_band):
                            temp_change_rmse[band_index] = np.linalg.norm(bands_fit_diff[IDsOld[sorted_index] - IDsOld[0], band_index]) / np.sqrt(num4rmse - CCDC_result[number_fitcurve - 2]['category'])

                        v_dif[:-1, :] = v_dif[1:, :]
                        v_dif[-1, :] = 0
                        v_dif_mag[:-1, :] = v_dif_mag[1:, :]
                        v_dif_mag[-1, :] = 0
                        vec_mag[:-1] = vec_mag[1:]
                        vec_mag[-1] = 0

                        for band_index in range(num_band):
                            v_dif_mag[-1, band_index] = y_arr[i_end + min_obs4stable, band_index] - autoTSPred(x_arr[i_end + min_obs4stable], fit_coeff[:, band_index])[0]
                            mini_rmse = max(adj_rmse[band_index], temp_change_rmse[band_index])
                            v_dif[-1, band_index] = v_dif_mag[-1, band_index] / mini_rmse
                        vec_mag[-1] = np.linalg.norm(v_dif[-1, :]) ** 2

                    if min(vec_mag) > Thr_change_detect:
                        CCDC_result[number_fitcurve - 2]['t_break'] = x_arr[i_end + 1]
                        CCDC_result[number_fitcurve - 2]['change_prob'] = 1
                        CCDC_result[number_fitcurve - 2]['magnitude'] = np.median(v_dif_mag, axis=0)

                        number_fitcurve += 1
                        i_start = i_end + 1
                        BL_train = 0

                    elif vec_mag[0] > Thrmax_change_detect:
                        x_arr = np.delete(x_arr, i_end + 1)
                        y_arr = np.delete(y_arr, i_end + 1, axis=0)
                        i_end -= 1
            i_end += 1

        # Two ways for processing the end of the time series
        if BL_train == 1:
            #  if no break find at the end of the time series
            #  define probability of change based on conse

            for i_conse in range(min_obs4stable - 1, -1, -1):
                if vec_mag[i_conse] <= Thr_change_detect:
                    id_last = i_conse
                    break

            CCDC_result[number_fitcurve - 2]['change_prob'] = (min_obs4stable - id_last) / min_obs4stable
            CCDC_result[number_fitcurve - 2]['t_end'] = x_arr[-min_obs4stable + id_last]

            if min_obs4stable > id_last:
                CCDC_result[number_fitcurve - 2]['t_break'] = x_arr[-min_obs4stable + id_last + 1]
                CCDC_result[number_fitcurve - 2]['magnitude'] = np.median(v_dif_mag[id_last: min_obs4stable, :], axis=0)

        elif BL_train == 0:

            #  2) if break find close to the end of the time series
            #  Use [min_obs4stable,min_num_c*n_times+min_obs4stable) to fit curve

            if number_fitcurve == record_fitcurve:
                i_start = 1
            else:
                i_start = np.where(x_arr >= CCDC_result[number_fitcurve - 2]['t_break'])[0][0]

            # Check if there is enough data
            if len(x_arr[i_start:]) >= min_obs4stable:
                # Define computed variables
                fit_cft = np.zeros((max_num_c, num_band))
                rmse = np.zeros(num_band)
                fit_diff = np.zeros((len(x_arr[i_start:]), num_band))
                qa = 20

                for band_index in range(num_band):
                    fit_cft[:, band_index], rmse[band_index], fit_diff[:, band_index] = autoTSFit(x_arr[i_start:], y_arr[i_start:, band_index], min_num_c)

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
    except:
        print(traceback.format_exc())
        print('Error')
        return []

    return CCDC_result

def plot_ccdc_segments(doy_arr, index_arr, CCDC_result, output_figpath, min_year, band_index=0, title="CCDC Fitting"):
    """
    按 CCDC_result 输出结果绘制拟合曲线与原始数据点
    """
    # 第一列是时间
    try:
        year_arr = doy_arr / 365.25 + min_year
        plt.figure(figsize=(20, 6))
        plt.scatter(doy_arr, index_arr, color='gray', s=10, label='Observed', alpha=0.5)

        colors = plt.cm.tab10(np.linspace(0, 1, len(CCDC_result)))

        for idx, segment in enumerate(CCDC_result):
            # 如果存在断点 t_break
            if segment['t_break'] > 0:
                t_start = segment['t_start']
                t_end = segment['t_end']
                coeffs = segment['coeffs'][:, band_index]
                # 构造预测用时间点
                t_range = np.linspace(t_start, t_end, t_end - t_start + 1)
                y_pred = autoTSPred(t_range, coeffs)
                label = f"Segment {idx+1}"
                plt.plot(t_range, y_pred, color=colors[idx], linewidth=2, label=label)
                plt.axvline(segment['t_break'], color=colors[idx], linestyle='--', alpha=0.6)

        # ===== 设置 xticks 为年份，单位仍是“天” =====
        min_doy = int(np.floor(np.min(doy_arr)))
        max_doy = int(np.ceil(np.max(doy_arr)))
        start_year = int(np.floor(min_doy / 365.25 + min_year))
        end_year = int(np.ceil(max_doy / 365.25 + min_year))

        # 计算每年结束对应的 DOY 值（x 轴刻度）
        year_tick_positions = [int((year - min_year) * 365.25 + 182.75) for year in range(start_year, end_year)]
        yline_positions = [int((year - min_year + 1) * 365.25 ) for year in range(start_year, end_year)]
        year_labels = [str(year) for year in range(start_year, end_year)]

        plt.xticks(ticks=year_tick_positions, labels=year_labels)
        for x in yline_positions:
            plt.axvline(x=x, color='lightgray', linestyle=':', linewidth=1)
        plt.axvline(x=0, color='lightgray', linestyle=':', linewidth=1)

        plt.xlabel("Year")
        plt.ylabel(f"Band {band_index + 1} Value")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_figpath, dpi=300)
        plt.close()
    except:
        pass

# if __name__ == '__main__':
#     doy_arr = [320,336,352,512,704,720,768,800,832,848,1040,1136,1536,1568,1840,2144,2160,2176,2208,2512,2528,2624,2688,2720,2944,3232,3344,3600,4160,4320,4736,4800,4880,5088,5096,5104,5200,5264,5688,5808,5848,5864,5880,5912,5944,6104,6152,6240,6288,6296,6312,6592,6616,6640,6648,6672,6880,6896,6912,7016,7032,7272,7288,7376,7384,7400,7536,7544,7616,7816,8072,8080,8096,8120,8176,8368,8384,8400,8424,8432,8496,8504,8696,8728,8816,8840,9088,9104,9120,9136,9248,9288,9408,9424,9432,9480,9560,9800,10144,10160,10208,10224,10240,10248,10320,10352,10368,10536,10544,10568,10576,10872,10896,10992,11216,11232,11272,11296,11320,11344,11408,11424,11440,11632,11680,11696,11744,11824,11848,11944,11992,12016,12120,12160,12384,12392,12416,12448,12456,12464,12496,12520,12576,12784,12792,12816,12832,12856,12864,12904,12912,13080,13120,13136,13144,13216,13240,13248,13272,13360,13368,13520,13544,13576,13628,13632,13650,13664,13728,13760,13770,13797,13808,13816,13824,13840,13856,13872]
#     trend_arr = [[32987.], [33094.], [33105.], [33492.], [33038.], [32995.], [33250.], [33225.], [33193.], [33364.], [32733.], [33312.], [33297.], [33458.], [33334.], [33574.], [32979.], [32982.], [33104.], [33270.], [32805.], [33218.], [33706.], [33905.], [32962.], [33108.], [33306.], [32948.], [33566.], [33100.], [33231.], [33607.], [33857.], [33382.], [33188.], [33372.], [33389.], [33502.], [33260.], [33156.], [33185.], [33059.], [33000.], [33136.], [33426.], [33153.], [33019.], [32776.], [33229.], [33308.], [33356.], [33146.], [32866.], [33456.], [33232.], [33587.], [33283.], [33320.], [33079.], [33379.], [33286.], [33224.], [32945.], [33311.], [33103.], [33504.], [34076.], [33333.], [33251.], [34055.], [32795.], [33245.], [33257.], [33514.], [33829.], [33324.], [32849.], [33475.], [33227.], [33079.], [33870.], [33848.], [33646.], [33360.], [34019.], [33944.], [33562.], [33754.], [33789.], [33262.], [34432.], [34888.], [34249.], [34251.], [33622.], [33824.], [34151.], [34037.], [34711.], [34604.], [34962.], [34788.], [34555.], [34428.], [35407.], [36021.], [36340.], [35207.], [35273.], [34657.], [34820.], [34528.], [35744.], [34739.], [34969.], [34520.], [34469.], [35456.], [33766.], [34860.], [36151.], [36538.], [36364.], [34924.], [34621.], [34849.], [34853.], [35284.], [35494.], [35069.], [35708.], [35134.], [34455.], [36033.], [35324.], [34713.], [34509.], [35218.], [34727.], [35595.], [35380.], [36213.], [35310.], [34156.], [33948.], [34312.], [34657.], [35076.], [35018.], [35634.], [35017.], [34422.], [34059.], [34147.], [34417.], [34905.], [35228.], [34956.], [34657.], [35229.], [34605.], [34469.], [34357.], [34599.], [35854.], [35047.], [35781.], [35986.], [34528.], [34738.], [34715.], [34821.], [34387.], [34770.], [35460.], [35046.], [34982.], [34923.]]
#     trend_arr = (np.array(trend_arr) - 32768) / 10000
#     ccdc_res = TrendSeasonalFit_v12_30Line(np.array(doy_arr), np.array(trend_arr))