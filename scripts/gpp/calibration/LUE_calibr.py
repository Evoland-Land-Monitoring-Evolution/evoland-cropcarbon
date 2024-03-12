""""
Script that will be used to calibrate the LUE for C3/C4 crops seperately.
"""

# import needed packages
from loguru import logger
import json
import os
import glob
import numpy as np
from pathlib import Path
import pandas as pd
from cropcarbon.gpp.constants import basefold_LUE
from cropcarbon.gpp.algorithm import get_GPP
from cropcarbon.gpp.calibration.methods import fit_LUE_param
from sklearn.model_selection import train_test_split
from cropcarbon.gpp.evaluation.metrics import get_val_metrics
from cropcarbon.utils.plotting import plot_cal_param



def gpp_per_field(GDMP_max_method, year, 
                  df_input, f_RUE,
                  settings,
                  field_id='siteid'): 
    """ 
    Wrapper function to calculate for each field 
    seperately the GPP
    """
    # for each field seperately calculate the GPP
    lst_GPP_sites = []
    for site in df_input[field_id].unique().tolist():
        logger.info(f"calculate GPP for {site} in {str(year)}")
        df_input_site = df_input.loc[df_input[field_id] == site]
        GPP, idx = get_GPP(GDMP_max_method, year,
                      df_input_site, f_RUE, 
                      settings)
        if idx is not None:
            # reindex the dataframe
            df_input_site = df_input_site.reindex(idx)
        df_input_site['GPP_model'] = GPP
        lst_GPP_sites.append(df_input_site)
    df_GPP_final = pd.concat(lst_GPP_sites)
    return df_GPP_final
        

def LUE_cal(settings, target_dir):

    #####################################
    ###       OUTPUT CONFIGURATION    ###
    #####################################
    outname_cal_param = f'CAL_LUE_{settings.get("VERSION_CAL")}.csv'
    outfolder_cal_param = os.path.join(target_dir, 'csv')
    outfolder_plot_param = os.path.join(target_dir, 'png')
    os.makedirs(outfolder_plot_param, exist_ok=True)
    os.makedirs(outfolder_cal_param, exist_ok=True)
    outdir_cal_param = os.path.join(outfolder_cal_param, outname_cal_param)

    if not settings.get('OVERWRITE') and os.path.exists(outdir_cal_param):
        return
    #####################################
    ####      INPUT PREPARATION     #####
    #####################################
    # Load first the prepared input data
    Basefold = settings.get('BASEFOLD_CAL_DATA')
    Version_input = settings.get('VERSION_INPUT')
    years_input = os.listdir(Path(Basefold).joinpath(Version_input))
    years_input = [item for item in years_input if not '.' in item]
    # Check if the GDMP max can be directly calculated
    # on a dekadal scale instead of daily
    GDMP_MAX_SCALE = settings.get('CAL_OPTIONS').get('GDMP_MAX_SCALE')
    # Check the scale of the input cal data that should be used
    GPP_scale = settings.get('CAL_OPTIONS').get('GPP_SCALE')
    years_exclude = settings.get('YEARS_EXCLUDE', None)
    if years_exclude is None:
        years_exclude = []

    # load the ancillary dataset containing information
    # on the type of land cover at each site
    df_anc = pd.read_csv(settings.get('ANC_FLUX_INFO'))

    
    #####################################
    ####      GPP CALCULATION       #####
    #####################################
    
    # set the LUE parameter to 1 such it has not yet an impact
    f_RUE = 1

    # list with the dataframes that will store
    # the modelled and target GPP
    lst_df_cal = []
    # for each year (different CO2 content) seperately determine
    # the GDMP with a LUE of 1
    for year in years_input:
        if year in years_exclude:
            continue
        logger.info('GETTING GPP WITH CONSTANT LUE '
                    f'FOR {str(year)}')
        # Load the corresponding input data
        files_input = glob.glob(os.path.join(Basefold,
                                             Version_input, 
                                             str(year), 'csv',
                                             f'*_{GPP_scale}_*.csv'))
        if len(files_input) == 0:
            raise ValueError(f'NO INPUT DATA FOR YEAR {str(year)}')
        # Concatenate them all
        df_input = pd.concat(pd.read_csv(f, index_col=0) for f in files_input)
        df_input_GPP= gpp_per_field(GDMP_MAX_SCALE, str(year),
                                    df_input, f_RUE, settings)
        lst_df_cal.append(df_input_GPP)

    # merge all dataframes together and
    # create array with target and modelled
    df_cal = pd.concat(lst_df_cal)

    # drop empty rows
    df_cal.dropna(axis=0, inplace=True)

    
    #####################################
    ####    PARAMETER CALIBRATION   #####
    #####################################
    # get the fitting method
    fit_method = settings.get('CAL_OPTIONS').get('CAL_METHOD')
    # get the split coefficient 
    split_coeff = settings.get("CAL_OPTIONS").get("CAL_SPLIT")
    val_spl = 1-split_coeff
    # Split the dataset based on the type of crop 
    # which requires a separate calibration
    lst_param_cal = []
    for Cr_type in settings.get('CROPTYPE_SPLIT').keys():
        logger.info(f'DOING CALIBRATION FOR {Cr_type} CROPS')
        # get the actual naming of the crop type based
        # on how it is called in the ancillary dataset
        Cr_name = settings.get('CROPTYPE_SPLIT').get(Cr_type)
        sites_filter = df_anc.loc[df_anc['igbp'] == Cr_name]['siteid']\
            .unique().tolist()
        df_cal_Cr = df_cal.loc[df_cal['siteid'].isin(sites_filter)]
        X_model = df_cal_Cr[['siteid', 'GPP_model']]
        Y_target = df_cal_Cr[['siteid', 'GPP_target']]
        # Split now the dataset in cal/val
        X_train, X_test, y_train, y_test = train_test_split(X_model, 
                                                            Y_target, 
                                                            test_size=val_spl, 
                                                            random_state=42)
        logger.info('START THE ACTUAL MODEL FITTING')
        LUE = fit_LUE_param(X_train, y_train, fit_method)
        # apply now the defined LUE on the test dataset
        GPP_pred = X_test['GPP_model'].values * LUE
        GPP_target = y_test['GPP_target'].values
        # get now the validation metrics
        dict_val_info = get_val_metrics(GPP_pred, GPP_target)
        # Add LUE to the dictionary
        dict_val_info.update({'LUE': np.round(LUE,2)})
        df_cal_info = pd.DataFrame(dict_val_info, index=[Cr_type])
        lst_param_cal.append(df_cal_info)

        # plot the corresponding result to gain some insights
        # on the potential bias in the GPP spectrum
        outname_plot = f'GPP_BIAS_TEST_DATASET_{Cr_type}.png'
        plot_cal_param(GPP_pred, GPP_target,
                       os.path.join(outfolder_plot_param, outname_plot),
                       dict_val_info, class_optim=Cr_type)
    df_cal_info_merged = pd.concat(lst_param_cal)
    df_cal_info_merged.to_csv(outdir_cal_param, index=True)

        
def main(basedir):
    # Get generic settings for preparing data
    settingsfile = str(basedir / 'Generic_settings.json')
    settings = json.load(open(settingsfile, 'r'))

    # define the target dir where the results will be stored
    target_dir = basedir.joinpath('results',
                                  settings.get('VERSION_CAL'))
    os.makedirs(target_dir, exist_ok=True)

    logger.info('Start calc GPP data on sites')
    LUE_cal(settings, target_dir)
    # write out the used settings for preparing the data
    meta_name = f'metadata_{settings.get("VERSION_CAL")}.json'
    if not os.path.exists(target_dir.joinpath(meta_name))\
        or settings.get("overwrite"):
        with open(target_dir / meta_name, 'w') as f:
            json.dump(settings, f, ensure_ascii=False)

    logger.success('FINISHED SUCESSFULLY')

if __name__ == "__main__":

    main(Path(basefold_LUE))