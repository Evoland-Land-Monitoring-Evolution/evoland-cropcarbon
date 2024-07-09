"""
UDF used to apply satio interpolation on the timeseries data
"""
import numpy as np
import sys
from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube
from openeo.udf import inspect


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply the satio interpolation on the timeseries data
    :param cube: XarrayDataCube
    :param context: Dict
    :return: XarrayDataCube
    """
    # first load the path from where the repo functions can be loaded
    sys.path.insert(0, 'tmp/venv_cropcarbon')
    sys.path.insert(0, 'tmp/venv_dep')
    from loguru import logger      
    from cropcarbon.utils.timeseries import to_satio_timeseries
    # load relevant context
    sensor = 'S2'
    ds = cube.get_array()
    # load the original dims
    orig_dims = ds.dims
    # reorder dimensions
    ds = ds.transpose('bands', 't', 'y', 'x')
    inspect(data=[ds.shape], message="UDF logging shape of my cube")
    # ensure that the data is in the proper format
    ds = ds.astype(np.float32)
    ts = to_satio_timeseries(ds, sensor)
    logger.info(f'Apply satio interpolation for {sensor}')
    ts = ts.interpolate()
    # convert this back to an xarray dataset
    xr_ts = ts.to_xarray()
    # create xarray similar as ds
    # create deep copy ds
    ds_interpol = ds.copy(deep=True)
    # add interpolated values to it
    ds_interpol.values = xr_ts.values
    # And make sure we revert back to original dimension order
    ds_interpol = ds_interpol.transpose(*orig_dims)
    return XarrayDataCube(ds_interpol)