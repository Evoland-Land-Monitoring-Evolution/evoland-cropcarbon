"""
Script with evaluation metrics to determine the perforamce
of the method
"""

import numpy as np
from sklearn.metrics import r2_score


def precision(MAE_out, RMSE_out):
    return RMSE_out**2 - MAE_out**2


def RMSE(pred, obs):
    return np.sqrt(((pred-obs)**2).mean())


def MAE(pred, obs):
    return np.round(np.mean(np.abs(pred-obs)), 2)


def get_val_metrics(pred, obs):
    dict_VAL = {}
    RMSE_out = np.round(RMSE(pred, obs),2)
    MAE_out = np.round(MAE(pred, obs),2)
    R2_out = np.round(r2_score(obs, pred),2)
    Prec_out = np.round(precision(MAE_out, RMSE_out), 2)

    dict_VAL.update({'RMSE': RMSE_out,
                     'MAE': MAE_out,
                     'PRECISION': Prec_out,
                     'R2_out': R2_out})
    return dict_VAL