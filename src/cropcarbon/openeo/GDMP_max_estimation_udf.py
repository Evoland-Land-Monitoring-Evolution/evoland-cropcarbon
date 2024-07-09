"""
UDF used to calculate the GDMP max based on the required input data
"""
import numpy as np
import sys
import xarray as xr
from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube


def _GDMP_max_grouped_data(group, f_LUE, METEO_SCALING):
    # Load the imported dependencies
    from cropcarbon.gpp.algorithm import GDMP_max
    
    # load all the input data for GDMP max calculation
    arr_tmin = group.sel(bands='temperature_min').values
    arr_tmin = arr_tmin * METEO_SCALING
    arr_tmax = group.sel(bands='temperature_max').values
    arr_tmax = arr_tmax * METEO_SCALING
    arr_rad = group.sel(bands='solar_radiation_flux').values
    # load the ECO2 factor
    arr_ECO2 = group.sel(bands='ECO2').values
    # get the LUE value for the specific land cover type
    if group.sel(bands='LC').values[0] in f_LUE.keys():
        f_LUE = f_LUE[group.sel(bands='LC').values[0]]
        arr_GDMPmax = GDMP_max(arr_tmin, arr_tmax, arr_rad,
                               f_RUE=f_LUE, f_Eco2=arr_ECO2)
    else:
        arr_GDMPmax = np.full(arr_tmin.shape, np.nan)
    # get name of dimension beside bands
    group_dims = list(group.dims)
    group_dims.remove('bands')
    # create xarray from GDMP max
    xr_GDMPmax = xr.DataArray(data=np.expand_dims(arr_GDMPmax, 0),
                              dims=group.dims,
                              coords={'bands': ['GDMP_max'],
                                      group_dims[0]:group.coords[group_dims[0]]}) # NOQA
    # concatenate the GDMP max to the group bands
    group = xr.concat([group, xr_GDMPmax], dim='bands')
    return group


def _apply_GDMP_max_LC(LC_dataset, ds, f_LUE_CROP,
                       f_LUE_GRAS, METEO_SCALING):
    """
    Function to calculate the GDMP max based on the land
    cover type (crop) and grassland
    This GDMP max will be calculated at daily scale.
    Args:
        LC_dataset (str): Defines the land cover dataset used
        ds (_type_): Xarray dataset with all required data
        f_LUE_CROP (_type_): LUE value for crop
        f_LUE_GRAS (_type_): LUE value for grassland
        METEO_SCALING (_type_): Scaling factor for meteo data
    Returns:
    GDMP max at daily scale based on LUE specific for crop and grassland
    """
    from loguru import logger
    # first apply only for the cropland locations the GDMP max calculation
    # filter meteo data for cropland locations
    if LC_dataset == "WORLDCOVER":
        CROP_CLASS = 40
        GRAS_CLASS = 30
        dict_LC_LUE = {CROP_CLASS: f_LUE_CROP,
                       GRAS_CLASS: f_LUE_GRAS}
    else:
        raise ValueError(f'Land cover dataset {LC_dataset} not supported!')

    # apply GDMP max function land cover specific
    logger.info(f'DICT FOR LUE is {dict_LC_LUE}')
    logger.info(f'METEO SCALING IS {METEO_SCALING}')
    logger.info(f'LC values looks like: {ds.sel(bands="LC").values}')
    ds_LC_specific = ds.groupby(ds.sel(bands='LC')).map(lambda x: _GDMP_max_grouped_data(x, dict_LC_LUE, METEO_SCALING)) # NOQA
    # get one band from ds
    ds_orig_band = ds.sel(bands='solar_radiation_flux')
    # drop band dimension to allow renaming it to the proper name
    ds_orig_band = ds_orig_band.drop('bands')
    ds_orig_band = ds_orig_band.expand_dims('bands')
    ds_orig_band = ds_orig_band.assign_coords(bands=['GDMP_max'])
    # now retrieve the GDMP max from the LC specific array
    ds_new_band = ds_LC_specific.sel(bands='GDMP_max')
    # align back to original xarray
    ds_new_band_aligned = xr.align(ds, ds_new_band, join='left')[1]
    return ds_new_band_aligned
    
    
def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply the GPP calculation on the timeseries data
    :param cube: XarrayDataCube
    :param context: Dict
    :return: XarrayDataCube
    """
    # first load the path from where the repo functions can be loaded
    sys.path.insert(0, 'tmp/venv_cropcarbon')
    sys.path.insert(0, 'tmp/venv_dep')
    from loguru import logger      
    # load relevant data
    ds = cube.get_array()
    # logger.info(f'DS LOOKS LIKE:{ds}')
    # logger.info(f'DS SHAPE LOOKS LIKE:{ds.shape}')
    # logger.info(f'SOLAR DATA LOOKS LIKE: {np.unique(ds.sel(bands="solar_radiation_flux"))}')
    # logger.info(f'ECO2 DATA LOOKS LIKE: {np.unique(ds.sel(bands="ECO2"))}')
    # logger.info(f'LC DATA LOOKS LIKE: {np.unique(ds.sel(bands="LC"))}')
    orig_dims = ds.dims
    # # drop bands from orig_dims
    orig_dims = list(orig_dims)
    orig_dims.remove('bands')
    # Load scale factor of meteo data
    METEO_SCALING = context.get('METEO_SCALING')
    f_LUE_CROP = context.get('f_LUE').get('CROP')
    f_LUE_GRAS = context.get('f_LUE').get('GRA')
    # get first the GDMP_max at daily scale
    logger.info('CALCULATE GPP MAX FIRST')
    # check which land cover dataset is used
    LC_dataset = context.get('LC_DATASET')
    # apply the GDMP max formula
    xr_GDMP_max = _apply_GDMP_max_LC(LC_dataset, ds,
                                     f_LUE_CROP, f_LUE_GRAS,
                                     METEO_SCALING)
    # revert to original dimensions
    ds_out = xr_GDMP_max.transpose(*orig_dims)
    return XarrayDataCube(ds_out)