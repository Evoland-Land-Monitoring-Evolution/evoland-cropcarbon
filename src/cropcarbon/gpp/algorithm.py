"""
In this code the functions for the GPP algorithm
will be stored
The algorithm consists of the following parameters:
GPP = R * Ec * fAPAR * ELUEc * Et * ECO2 x Eres * Econv
With:
GPP: Gross Primary Production [gC/m²/dekad]
R: Total shortwave incoming radiation [GJt/ha/day]
Ec: Fraction of PAR in total shortwave (0.48) [Jp/Jt]
fAPAR: PAR fraction absorbed by green vegetation (0-1) [Jap/Jt]
ELUEc: Light use efficiency at optimum: kgDM/GJap 
Et: Normalized temperature effect (0-1)
ECO2: Normalized CO2 fertilization effect (0-1)
Eres: Fraction kept after omitted effects (1)
Econv: Conversion factor from GDMP to GPP (0.05)

"""

# import needed packages
import numpy as np
import pandas as pd
from cropcarbon.utils.timeseries import aggregate_TS
from loguru import logger


def Et(T12):
    """
    Function that will calculate the normalized temperature effect throught time
    (temperature response curve). The only required input is the the temperature 
    at noon. 
    
    :param Td: Temperature at noon [K]
    :return: the value of pTd.
    """

    # TODO check if not directly the daily mean could be used from AGERA5
    # instead of deriving it from the min and maximum temperature 
    
    # CONSTANT PARAMETERS
    # constant
    c1 = 21.77
    # Activation energy  
    HaP = 52750
    # Universal gas constant
    Rg = 8.3144
    # Entropy of the denaturation equilibrium of CO2
    DeltaS = 704.98
    # Deactivation energy 
    HdP = 211000

    # CALCULATION
    pTd_numerator = np.exp(c1 - (HaP / (Rg * T12)))
    pTd_denominator = 1 + np.exp(((DeltaS * T12) - HdP) / (Rg * T12))
    pTd = pTd_numerator / pTd_denominator

    # RESULTS
    return pTd


def get_CO2_conc(year: int, dir_CO2_content: str):
    df_CO2 = pd.read_csv(dir_CO2_content)
    # filter on the year
    df_CO2_year = df_CO2.loc[df_CO2.year == year]
    # take just the mean value within the year 
    CO2_conc = df_CO2_year.co2.mean()
    return CO2_conc


def Eco2(CO2conc, T12):
    """"
    Code to determine the CO2 fertilization effect, which 
    is driven by the temperature and the CO concentration.

    :param CO2conc: Actual CO2 concentration of the specific year [PPMv]
    :param T12: Temperature at noon [K]
    """

    # TODO check if not directly the daily mean could be used from AGERA5
    # instead of deriving it from the min and maximum temperature 

    # CONSTANTS
    O2conc = 20.9
    CO2concref = 281
    Rg = 8.31
    Ea1 = 59400
    A1 = 2.419 * (10 ** 13)
    Ea2 = 109600
    A2 = 1.976 * (10 ** 22)
    E0 = 13913.5
    A0 = 8240
    Et = -42896.9
    At = 7.87 * (10 ** -5)

    # CALCULATION
    T1arrix = np.where(T12 >= 288.13)
    T2arrix = np.where(T12 < 288.13)
    if len(T12.shape) == 1:
        lines = len(T12)
        samples = 0
        Km = np.zeros((lines), 'float')
    elif len(T12.shape) == 2:
        samples = len(T12[0])
        lines = len(T12)
        Km = np.zeros((lines, samples), 'float')
    else:
        samples_1 = T12.shape[1]
        samples_2 = T12.shape[2]
        lines = T12.shape[0]
        Km = np.zeros((lines, samples_1, samples_2), 'float')

    Km[T1arrix] = A1 * np.exp(-Ea1 / (Rg * T12[T1arrix]))
    Km[T2arrix] = A2 * np.exp(-Ea2 / (Rg * T12[T2arrix]))

    K0 = A0 * np.exp(-E0 / (Rg * T12))
    t = At * np.exp(-Et / (Rg * T12))

    Nco2FE_numerator1 = CO2conc - (O2conc / (2 * t))
    Nco2FE_numerator2 = (Km * (1 + (O2conc / K0))) + CO2concref
    Nco2FE_denominator1 = CO2concref - (O2conc / (2 * t))
    Nco2FE_denominator2 = (Km * (1 + (O2conc / K0)) + CO2conc)

    Nco2FE = (Nco2FE_numerator1 / Nco2FE_denominator1) * \
             (Nco2FE_numerator2 / Nco2FE_denominator2)

    # RESULTS
    return Nco2FE


def GDMPtoDMP(GDMP, DMPfrac=0.5):
    # CONSTANTS
    DMP = GDMP * DMPfrac
    return DMP


def GDMP_max(arr_tmin_k, arr_tmax_k, arr_rad,
             f_CO2conc=405, f_RUE=2.54, 
             f_Ec=0.48, f_Eco2=None):
    '''
    General note    : all arrays should represent the same spatial grid
    arr_tmin        : Array containing the daily minimum values for each grid cell [K]
    arr_tmax        : Array containing the daily maximum values for each grid cell [K]
    arr_rad         : Array containing shortwave incoming solar radiation for each grid cell [j/m2]
    CO2conc         : Constant value representing the atmospheric CO2 concentration of the period
    RUE             : Radiation Use Efficiency (expressed as kgDM/GJ PAR]
    f_Ec            : Fraction of PAR in total shortwave
    f_Eco2          : ECO2 factor if calc_ECO2 is False
    '''

    # Rescaling arrays:
    arr_t12_k = (0.25 * arr_tmin_k) + (0.75 * arr_tmax_k)  # Calculate Tday
    arr_rad_u = arr_rad / 10**5  # to convert from j/m2 to GJ/ha

    # GDMP max theoretical max without fAPAR info

    # get first the value of each parameter needed
    f_Et = Et(arr_t12_k)
    if f_Eco2 is None:
        f_Eco2 = Eco2(f_CO2conc, arr_t12_k)
    GDMP_max = f_Ec * f_RUE * f_Eco2 * f_Et * arr_rad_u

    return GDMP_max


def GDMP_max_to_GDMP(arr_GDMP_max, arr_fapar,
                     unit=1, GDMPorDMP='DMP', 
                     f_DMPfraction=0.5):
    '''
    General note    : all arrays should represent the same spatial grid
    arr_fapar       : Array containing dekadal fAPAR values 
    arr_GDMP_max    : Array containing calculated GDMP max values in kg/DM/ha_day
    unit            : 1= kg DM/ha, 2 = g C/m2
    GDMPorDMP       : Output should be either GDMP ('GDMP') or DMP ('DMP)
    DMPfraction     : DMPfraction used to convert GDMP into DMP (DMP = GDMP * DMPfraction). Use DMPfraction=1 to get GDMP outputs
    '''
    arr_GDMP = arr_GDMP_max * arr_fapar

    if unit == 2:
        arr_GDMP = arr_GDMP * 0.5 * 0.1

    if GDMPorDMP == 'GDMP':
        return arr_GDMP
    elif GDMPorDMP == 'DMP':
        arr_DMPmax = GDMPtoDMP(arr_GDMP, 
                               f_DMPfraction)
        return arr_DMPmax


def GDMP(arr_fapar, arr_tmin, arr_tmax, arr_rad,
         f_CO2conc=405, f_RUE=2.54, unit=1,
         GDMPorDMP='DMP', f_DMPfraction=0.5,
         f_Ec=0.48):
    '''
    General note    : all arrays should represent the same spatial grid
    arr_fapar       : Array containing dekadal fAPAR values 
    arr_tmin        : Array containing the daily minimum values for each grid cell [K]
    arr_tmax        : Array containing the daily maximum values for each grid cell [K]
    arr_rad         : Array containing shortwave incoming solar radiation for each grid cell [j/m2]
    CO2conc         : Constant value representing the atmospheric CO2 concentration of the period
    RUE             : Radiation Use Efficiency (expressed as kgDM/GJ PAR]
    unit            : 1= kg DM/ha, 2 = g C/m2
    GDMPorDMP       : Output should be either GDMP ('GDMP') or DMP ('DMP)
    DMPfraction     : DMPfraction used to convert GDMP into DMP (DMP = GDMP * DMPfraction). Use DMPfraction=1 to get GDMP outputs
    f_Ec              : Fraction of PAR in total shortwave
    '''

    # get first the max GDMP
    arr_GDMPmax = GDMP_max(arr_tmin, arr_tmax,
                           arr_rad, f_CO2conc=f_CO2conc,
                           f_RUE=f_RUE,
                           f_Ec=f_Ec)
    arr_GDMP = arr_GDMPmax * arr_fapar
    if unit == 2:
        arr_GDMP = arr_GDMP * 0.5 * 0.1

    if GDMPorDMP == 'GDMP':
        return arr_GDMP
    elif GDMPorDMP == 'DMP':
        arr_DMPmax = GDMPtoDMP(arr_GDMP, 
                               f_DMPfraction)
        return arr_DMPmax


def get_GPP(GDMP_max_method, year, 
            df_input, f_RUE,
            settings):
    df_input.index = pd.to_datetime(df_input.index)
    # load the input variables for the GPP calculation
    FAPAR_type = settings.get('CAL_OPTIONS').get('FAPAR_TYPE') 
    arr_tmin = df_input['temperature_min'].values
    arr_tmax = df_input['temperature_max'].values
    arr_rad = df_input['solar_radiation_flux'].values
    # Get the valid CO2 conc for this year
    if settings.get('CO2_CONC_METHOD') == "YEAR":
        dir_CO2_info = settings.get('CO2_CONC_INFO')
        CO2_conc = get_CO2_conc(int(year), dir_CO2_info)
    else:
        CO2_conc = 405
    if GDMP_max_method == 'DAY':

        # first calculate the GDMP MAX at daily scale
        GDMP_max_arr = GDMP_max(arr_tmin, arr_tmax,
                                arr_rad, CO2_conc,
                                f_RUE)
        # convert to dataframe and aggregate
        df_GDMP_max = pd.DataFrame(GDMP_max_arr, 
                                   index=df_input.index)
        df_GDMP_max.index = pd.to_datetime(df_GDMP_max.index)
        # filter the GDMP max on these positions
        df_GDMP_max_composite = aggregate_TS(df_GDMP_max,
                                              df_GDMP_max.index[0],
                                              df_GDMP_max.index[-1],
                                              settings.get('CAL_OPTIONS')
                                              .get('GPP_SCALE').lower(),
                                              'mean')
        df_FAPAR_site = df_input[FAPAR_type]
        df_FAPAR_site = df_FAPAR_site.reindex(df_GDMP_max_composite.index)
        # store the new index for later on reindexing
        idx = df_FAPAR_site.index
        GDMP_max_arr = np.squeeze(df_GDMP_max_composite.values,
                                  1)
        arr_FAPAR = df_FAPAR_site.values
        arr_FAPAR = np.reshape(arr_FAPAR, (len(arr_FAPAR)))

        # now convert the GDMP max to GDMP by combining with FAPAR
        GDMP_arr = GDMP_max_to_GDMP(GDMP_max_arr, arr_FAPAR,
                                    unit=2, GDMPorDMP='GDMP')
        return GDMP_arr, idx
    elif GDMP_max_method == 'DEKAD':
        logger.info(f'START GPP CALCULATION FOR YEAR: {str(year)}')
        # load the required input variables for the calculation
        arr_FAPAR = df_input[FAPAR_type].values
        GDMP_arr = GDMP(arr_FAPAR, arr_tmin, arr_tmax,
                        arr_rad, f_CO2conc=CO2_conc,
                        f_RUE=f_RUE, unit=2,
                        GDMPorDMP='GDMP')
    else:
        raise ValueError(f'GDMP MAX CALCULATION AT {GDMP_max_method}'
                             'SCALE NOT SUPPORTED!!')
    return GDMP_arr, None
