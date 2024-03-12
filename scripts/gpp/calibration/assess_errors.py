"""
In this script an assement after the calibration will be conducted to define
in which situations the model is not yet able to properly represent the actual 
GPP. For this some variables will be selected to relate it 
with the error in GPP. 
"""

# import needed packages
import json
import os
from pathlib import Path
import pandas as pd
from cropcarbon.gpp.constants import basefold_cal
from loguru import logger
import glob


def run_asssessment(basedir, settings):
    logger.info('START PREPARING DATA FOR ASSESSMENT')

    # load the data first per defined split 
    SPLIT_INFO  = settings.get('CAL').get('CROPTYPE_SPLIT')
    # load the dataframe with ancillaty information per site
    df_anc = pd.read_csv(settings.get('CAL').get('ANC_FLUX_INFO'))
    # check if certain years should be exclude
    years_exclude = settings.get('YEARS_EXCLUDE', None)
    if years_exclude is None:
        years_exclude = []
    # dictionary that will store per split class 
    # the corresponding data for assessment
    dict_GPP_split = {}
    Basefold_input = settings.get('CAL').get('BASEFOLD_CAL_DATA')
    VERSION_INPUT = settings.get('CAL').get('VERSION_INPUT')
    years_input = os.listdir(Path(Basefold_input).joinpath(VERSION_INPUT))
    years_input = [item for item in years_input if not '.' in item]
    GPP_scale = settings.get('CAL').get('CAL_OPTIONS').get('GPP_SCALE')

    for Cr_type in SPLIT_INFO.keys():
        for year in years_input:
            if year in years_exclude:
                continue
            logger.info(f'PREPARE DATA FOR {Cr_type} CROPS')
            # get the actual naming of the crop type based
            # on how it is called in the ancillary dataset
            Cr_name = SPLIT_INFO.get(Cr_type)
            sites_filter = df_anc.loc[df_anc['igbp'] == Cr_name]['siteid']\
                .unique().tolist()
            # load now the input data for the corresponding sites
            # Load the corresponding input data
            files_input = glob.glob(os.path.join(Basefold_input,
                                                 VERSION_INPUT, 
                                                 str(year), 'csv',
                                                 f'*_{GPP_scale}_*.csv'))
            # filter now only on the relevant sites
            files_input = [item for item in files_input]

    



def main(basedir):
    # Get generic settings for preparing data
    settingsfile = str(basedir / 'Generic_settings.json')
    settings = json.load(open(settingsfile, 'r'))

    # add calibration settings
    calfile = str(Path(basedir).parent / settings.get('PARAM') / 'results'
                  / settings.get('VERSION_CAL') / 
                  f'metadata_{settings.get("VERSION_CAL")}.json')
    cal_settings = json.load(open(calfile, 'r'))

    settings.update({'CAL': cal_settings})

    logger.info('Starting GPP assessment ...')
    run_asssessment(basedir, settings)

    # write out the used settings for preparing the data
    target_dir = basedir.joinpath('data',
                                  settings.get('VERSION'))
    meta_name = f'metadata_{settings.get("VERSION")}.json'
    if not os.path.exists(target_dir.joinpath(meta_name))\
        or settings.get("overwrite"):
        with open(target_dir / meta_name, 'w') as f:
                json.dump(settings, f, ensure_ascii=False)

    logger.success('FINISHED SUCESSFULLY')

if __name__ == "__main__":
    meta_dir = os.path.join(basefold_cal, 'assessment')
    main(Path(meta_dir))
