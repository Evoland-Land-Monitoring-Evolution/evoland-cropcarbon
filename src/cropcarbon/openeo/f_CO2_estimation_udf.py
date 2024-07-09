"""
UDF that will be used to estimate the CO2 coefficient for GDMP max estimation.
This factor depends on the time dimension and hence a different UDF should
be used before the actual GDMP calculation.
"""

import numpy as np
import sys
import xarray as xr
from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube


def get_CO2_grouped_data(group, METEO_SCALING):
    
    """
    Get the CO2 factor for GDMP max calculation
    for the group

    Args:
        group (_type_): grouped xarray based on time
        METEO_SCALING (_type_): Scaling factor for meteo data
    """
    from cropcarbon.gpp.algorithm import Eco2
    
    # below a dictionary with for every year (2017-2022)
    # the average CO2 concentration
    # the origins from CSV file with monthly CO2 concentration
    dict_CO2_yr = {
        2017: 406.52,
        2018: 408.53,
        2019: 411.42,
        2020: 413.95,
        2021: 416.11,
        2022: 418.22}
    # get the year for the group
    year = group.t.values.astype('datetime64[Y]').astype(int) + 1970
    year = np.unique(year)[0]
    # load the required input data for the CO2 concentration
    arr_tmin = group.sel(bands='temperature_min').values
    arr_tmin = arr_tmin * METEO_SCALING
    arr_tmax = group.sel(bands='temperature_max').values
    arr_tmax = arr_tmax * METEO_SCALING
    # Calculate the midday temperature:
    arr_t12_k = (0.25 * arr_tmin) + (0.75 * arr_tmax)  # Calculate Tday
    
    # get the CO2 concentration for this year
    if year in dict_CO2_yr.keys():
        CO2_conc = dict_CO2_yr[year]
    else:
        # derive the CO2 concentration from the closest year
        prev_CO2_conc = dict_CO2_yr[min(dict_CO2_yr.keys(),
                                        key=lambda x: abs(x-year))]
        # get the difference in years to the closest year
        diff_yr = year - min(dict_CO2_yr.keys(),
                             key=lambda x: abs(x-year))
        # derive the expected CO2 concentration
        CO2_conc = prev_CO2_conc + (2.2*diff_yr)
        
    # calculate the Eco2 factor       
    arr_Eco2 = Eco2(CO2_conc, arr_t12_k)
    # get name of dimension beside bands
    group_dims = list(group.dims)
    group_dims.remove('bands')

    # create xarray from ECO2
    coords_new_xr = group.drop_vars('bands').coords
    xr_ECO2 = xr.DataArray(data=arr_Eco2,
                           dims=group_dims,
                           coords=coords_new_xr) # NOQA
    xr_ECO2 = xr_ECO2.expand_dims('bands')
    xr_ECO2 = xr_ECO2.assign_coords(bands=['ECO2'])
    # concatenate the GDMP max to the group bands
    group = xr.concat([group, xr_ECO2], dim='bands')
    return group
        

def _get_CO2_factor(ds, METEO_SCALING):
    
    """
    Function that will define the year dependent CO2 factor

    Args:
        ds (_type_): xaarray dataset with all required data
        METEO_SCALING (_type_): Scaling factor for meteo data
    """

    # apply a groupby on the time dimension
    ds_CO2_year_specific = ds.groupby(ds.t.dt.year).map(lambda x: get_CO2_grouped_data(x, METEO_SCALING)) # NOQA
    
    return ds_CO2_year_specific


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
    orig_dims = ds.dims
    # Load scale factor of meteo data
    METEO_SCALING = context.get('METEO_SCALING')
    # get first the GDMP_max at daily scale
    logger.info('CALCULATE ECO2 FACTOR FOR GDMP MAX')
    # apply the GDMP max formula
    xr_ECO2 = _get_CO2_factor(ds, METEO_SCALING)
    # revert to original dimensions
    ds_out = xr_ECO2.transpose(*orig_dims)
    return XarrayDataCube(ds_out)