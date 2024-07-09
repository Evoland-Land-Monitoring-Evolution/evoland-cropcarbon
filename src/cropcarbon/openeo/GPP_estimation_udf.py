"""
UDF to calculate based on the GDMP max and the FAPAR, the final GPP
"""

import sys
import xarray as xr
import numpy as np
from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube


def _GDMPmax_to_GDMP(ds):
    """
    Function to convert the GDMP max to GDMP
    Args:
        ds (_type_): Xarray dataset with all required data
    Returns:
    GDMP at daily scale based on LUE specific for crop and grassland
    """
    # Load the imported dependencies
    from cropcarbon.gpp.algorithm import GDMP_max_to_GDMP
    # first apply only for the cropland locations the GDMP max calculation
    # filter meteo data for cropland locations
    arr_FAPAR = ds.sel(bands='FAPAR').values
    arr_GDMPmax = ds.sel(bands='GDMP_max').values
    arr_GPP = GDMP_max_to_GDMP(arr_GDMPmax, arr_FAPAR,
                               unit=2, GDMPorDMP='GDMP')
    # get the orig coords from the ds
    ds_no_band = ds.drop('bands')
    # create xarray from GDMP max
    dict_coords = {'bands': ['GPP']}
    dict_coords.update(ds_no_band.coords)
    xr_GPP = xr.DataArray(data=np.expand_dims(arr_GPP, 0),
                          dims=ds.dims,
                          coords=dict_coords)
    # concatenate the GDMP max to the group bands
    ds = xr.concat([ds, xr_GPP], dim='bands')
    return ds


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply the UDF to the datacube
    Args:
        cube (XarrayDataCube): The input datacube
        context (Dict): The context of the UDF
    Returns:
    XarrayDataCube with the GDMP estimation
    """
    # first load the path from where the repo functions can be loaded
    sys.path.insert(0, 'tmp/venv_cropcarbon')
    sys.path.insert(0, 'tmp/venv_dep')
    from loguru import logger      

    # load the dataset
    ds = cube.get_array()
    # load the original dimensions
    orig_dims = ds.dims
    # drop bands from orig_dims
    orig_dims = list(orig_dims)
    orig_dims.remove('bands')
    # apply the GDMP max to GDMP conversion
    logger.info('START CONVERSION GDMP max to GPP')
    ds = _GDMPmax_to_GDMP(ds)
    # keep only the GPP band
    ds = ds.sel(bands='GPP')
    # revert to original dims
    ds = ds.transpose(*orig_dims)
    logger.info('GPP calculation succeeded')
    
    return XarrayDataCube(ds)