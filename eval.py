import numpy as np


def mask_np(array, null_val):
    if np.isnan(null_val):
        mask = ~np.isnan(array)
    else:
        mask = np.not_equal(array, null_val)
    return mask.astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan, threshold=5.0):

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        # drop targets that are effectively zero to avoid exploding percentage errors
        valid = np.abs(y_true) >= threshold
        mask = mask * valid.astype('float32')

        # check if there are valid elements
        if mask.sum() == 0:
            return 0.0

        # avoid inf*0 -> nan by masking the denominator before division
        denom = np.where(mask, y_true, 1.0)
        mape = np.abs((y_pred - y_true) / denom)
        masked_mape = np.nan_to_num(mape * mask, nan=0.0, posinf=0.0, neginf=0.0)

        # return average of valid elements only
        return (masked_mape.sum() / mask.sum()) * 100


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    if mask.sum() == 0:
        return 0.0
    mse = (y_true - y_pred) ** 2
    masked_mse = mse * mask
    return np.sqrt(masked_mse.sum() / mask.sum())


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    if mask.sum() == 0:
        return 0.0
    mae = np.abs(y_true - y_pred)
    masked_mae = mae * mask
    return masked_mae.sum() / mask.sum()
