"""
Function that will assist in the calculation of some specific indices
"""

# import needed packages
import numpy as np
import pandas as pd

def calc_ESI(data):
    "Function to calculate the evaporative stress index"
    ESI = (data['e'] / data['pev']).values
    # clip values outside the range to the boundaries [0,1]
    ESI = np.clip(ESI, 0, 1)
    df_ESI = pd.DataFrame(ESI, index=data.index)
    return df_ESI

def calc_VPD(data, es0=0.611):
    """
    calculate vapour pressure deficit in PA based on method consutlable here:
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2016JD025855.
    Other method option if turned out needed can be found here: 
    https://gis.stackexchange.com/questions/436605/vapour-pressure-deficit-vpd-calculation-from-era5-google-earth-engine
    """

    # convert the temperature in Kelvin to Degrees
    T_max = data['temperature_max'] - 273.15
    T_min = data['temperature_min'] - 273.15
    tmaxmin = (8.635 * (T_max + T_min))/(0.5*(T_max + T_min) + 237.3)
    vpd = (es0 * np.exp(tmaxmin)-(0.10*data['vapour_pressure'])) * 1000
    # clip values to the range of [0, 10000 Pa]
    vpd = np.clip(vpd, 0, 10000)
    df_vpd = pd.DataFrame(vpd.values, index=data.index)
    return df_vpd