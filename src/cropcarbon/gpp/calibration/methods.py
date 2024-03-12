"""
Methods that can be used for calibration
"""

# import needed packages
import numpy as np 
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import pandas as pd

def fit_LUE_param(xarr, yarr, method, 
                  param_target='GPP'):

    if method == 'NON_LIN_LSQF':
        def func(x, p2):
            return x * p2
        # keep only the values related
        # to the target parameter
        xarr_clean = np.squeeze(xarr[[f'{param_target}_model']].values,
                                axis=1)
        yarr_clean = np.squeeze(yarr[[f'{param_target}_target']].values,
                                axis=1)

        popt, _ = curve_fit(func, xarr_clean, 
                               yarr_clean, p0=(2.54))
        LUE = popt[0]
        return LUE
    elif method == 'MXD_MODEL':
        yarr[f"{param_target}_target_log"] = np.log(yarr[f"{param_target}"
                                                         "_target"])
        xarr[f"{param_target}_model_log"] = np.log(xarr[f"{param_target}"
                                                        "_model"])
        # specify the random effects in this case the site id
        # the fixed effect is in this case the C3/C4 subdivision
        vc = {'siteid': '0 + C(siteid)'}
        df_merge = pd.concat([xarr, yarr], axis=1)
        df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()].copy()
        # drop index column
        df_merge = df_merge.reset_index(drop=True)
        # create just one group in this case 
        # for the fixed effect
        df_merge['GROUP'] = ['SAME'] * df_merge.shape[0]
        formula = f"{param_target}_target_log~ {param_target}_model_log"
        md = smf.mixedlm(formula, 
                         df_merge, vc_formula=vc,
                         groups="GROUP", re_formula="1")
        mdf = md.fit(method=["lbfgs"])
        # get the calibrated LUE
        LUE = np.exp(mdf.params[0])
        return LUE
    else:
        raise ValueError(f'{method} not yet supported!!')