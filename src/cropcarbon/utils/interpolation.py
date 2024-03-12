"""
Interpolation functions that could be applied to fill-in 
the gaps on timeseries. 
"""

# import needed packages
from loguru import logger
from cropcarbon.utils.timeseries import to_satio_timeseries
import pandas as pd
import xarray as xr
import numpy as np

def satio_interpolation(df: pd.DataFrame, sensor: str):
    # convert first to xarray
    band_names = list(df.columns)
    timestamps = list(df.index.values)

    ds = xr.DataArray(
        data=np.reshape(df.values[:,0],(1,len(timestamps),1,1)),
        dims=('bands', 't', 'y', 'x'),
        coords={
            'bands': band_names,
            't': timestamps,
            'y': [0],
            'x': [0]
        }
    )
    # ensure that the data is in the proper format
    ds = ds.astype(np.float32)
    
    ts = to_satio_timeseries(ds, sensor)
    logger.info(f'Apply satio interpolation for {sensor}')
    ts = ts.interpolate()

    # now convert back to the original dataframe
    df_interpolated = pd.DataFrame(np.reshape(ts.data, (len(timestamps), 
                                                        len(band_names)))
                                   , index = timestamps,columns= band_names)

    return df_interpolated
