""""
Function that will prepare the inputs to allow the calibration of the GPP model.
This includes for example, a proper pre-processing of the data.
"""

from loguru import logger
import pandas as pd
import os
import glob
import numpy as np
import datetime

from cropcarbon.utils.timeseries import aggregate_TS
from cropcarbon.utils.interpolation import satio_interpolation
from cropcarbon.utils.plotting import plot_cal_TS

def compile_ref(df, priority_var = {1: 'GPP_NT_VUT_MEAN', 
                                    2: 'GPP_NT_CUT_MEAN'}):
    # check first if values from 
    # the priority col are present
    if priority_var.get(1) in df.columns:   
        df_col1 = df[[priority_var.get(1)]]
    else:
        df_col1 = None

    if priority_var.get(2) in df.columns:   
        df_col2 = df[[priority_var.get(2)]]
    else:
        df_col2 = None
    
    if df_col1 is None:
        df_compiled = df_col2
        df_compiled.columns = ['GPP_target']
        # drop empty dates:
        df_compiled.dropna(inplace=True)
        return df_compiled
    # check if there are empty rows in the priority variable
    empty_rows = np.where(pd.isnull(df_col1))
    
    # now start filling in this empty rows 
    # with the second priority variable
    for empty_row in empty_rows[0]:
        if not df_col2.iloc[empty_row].isnull().any():
            df_col1.iloc[empty_row] = df_col2.iloc[empty_row]
    df_compiled = df_col1
    df_compiled.columns = ['GPP_target']
    # drop empty dates:
    df_compiled.dropna(inplace=True)
    return df_compiled
    

def compile_data_calibration(site, settings, 
                             temp_scale, year):
    logger.info(f'START DATA PREPARATION FOR: {site}')

    # load the required datasets
    DATASETS = list(settings.get('DATASETS').keys())

    # folder in which the reference data is stored
    folder_ref = os.path.join(settings.get('REF_FOLDER'),
                               'csv', f'gpp_{temp_scale.lower()}', 
                               'grouped')
   
    start = f'{str(year)}-01-01'
    end = f'{str(year)}-12-31'

    # load the reference in-situ data
    file_ref = glob.glob(os.path.join(folder_ref, f'*_{str(year)}_cleaned.csv'))
    if len(file_ref) == 0:
        raise Exception(f'NO REFERENCE DATA AVAILABLE FOR YEAR {year}')
    df_ref = pd.read_csv(file_ref[0], index_col=0)
    # filter on the site of consideration
    df_ref = df_ref.loc[df_ref['siteid'] == site]

    # if no reference data available for 
    # this specific site, just skip it
    if df_ref.empty:
        logger.info(f'NO DATA FOR: {site} in {str(year)}')
        return None
    # ensure index is in datetime date format
    df_ref.index = pd.to_datetime(df_ref.index).date

    # compile now the ref to keep only the relevant info
    df_ref_compiled = compile_ref(df_ref)

    lst_df_datasets = []
    lst_df_datasets.append(df_ref_compiled)
    for dataset in DATASETS:
        # load the corresponding extracted data (if available)
        folder_data = os.path.join(settings.get('EXTRACTION_FOLDER'),
                                    site, 'extractions', dataset, year)
        file_data = glob.glob(os.path.join(folder_data, '*.csv'))
        if len(file_data) == 0:
            continue
        if len(file_data) > 1:
            raise Exception(f'Multiple extracted files for {site}')
        data = pd.read_csv(file_data[0], index_col=0)

        # rescale the data if needed
        scale = settings.get('DATASETS').get(dataset).get('SCALE')
        if scale is None:
            logger.warning(f'APPLY NO SCALING FOR {dataset}')
            scale = 1
        data = data * scale
        if dataset == 'AGERA5':
            # do not appy scaling at solar radiation
            data['solar_radiation_flux'] = data['solar_radiation_flux']/scale
            # apply a further scaling for vapour pressure
            data['vapour_pressure'] = data['vapour_pressure'] * scale*10
            from cropcarbon.utils.indices import calc_VPD
            data['VPD'] = calc_VPD(data)
        if dataset == "ET":
            # Need to calculate the evaporative stress index
            from cropcarbon.utils.indices import calc_ESI
            data['ESI'] = calc_ESI(data)
            data = data[['ESI']]
        # ensure the index is in datetime format
        data.index = pd.to_datetime(data.index).date

        # apply a temporal aggregation of the data 
        # if different from the composite scale
        if settings.get('DATASETS').get(dataset).get('COMPOSITED_METHOD', None) != temp_scale \
            and settings.get('DATASETS').get(dataset).get('COMPOSITING'):
            reducer = settings.get('DATASETS').get(dataset).get('COMPOSITE_REDUCER', None)
            if reducer is None:
                logger.warning('Took default reducer for composting timeseries')
                reducer = 'mean'
            # check if on this dataset some interpolation should be done 
            # in that case the start and end should be taken from the range
            # in the dataframe
            if settings.get('DATASETS').get(dataset).get('interpolation', None) is None:
                data_aggreg = aggregate_TS(data, start, end, 
                                        temp_scale.lower(), reducer)
            else:
                data_aggreg = aggregate_TS(data, data.index.values[0], 
                                            data.index.values[-1], 
                                            temp_scale.lower(), reducer)
        else:
            data_aggreg = data.loc[datetime.datetime.strptime(start, '%Y-%m-%d').date(): 
                                    datetime.datetime.strptime(end, '%Y-%m-%d').date()]
        
        # check if an interpolation should be done
        if settings.get('DATASETS').get(dataset).get('interpolation'):
            interpol_method = settings.get('DATASETS')\
                                .get(dataset).get('interpolation')
            if interpol_method == "satio":
                sensor = dataset.split('_')[0]
                data_interpol = satio_interpolation(data_aggreg, sensor)
                # clip the data to the desired period
                data_final = data_interpol.loc[datetime.datetime.strptime(start, '%Y-%m-%d').date(): 
                                             datetime.datetime.strptime(end, '%Y-%m-%d').date()]
            else:
                raise ValueError(f'INTERPOLATION METHOD {interpol_method} NOT YET SUPPORTED')
        else:
            data_final = data_aggreg

        # drop empty dates:
        data_final.dropna(inplace=True)

        if dataset == 'CROPSAR':
            data_final.columns = ['CROPSAR']

        # append the dataframe to the list
        lst_df_datasets.append(data_final)

    # join now all the data together in a single dataframe
    df_compiled = pd.concat(lst_df_datasets, axis=1)
    df_compiled['siteid'] = [site] * df_compiled.shape[0]
    df_compiled = df_compiled.sort_index()
    logger.info(f'FINISHED DATA PREPARATION FOR: {site}')

    return df_compiled

            
def main(basedir, settings):
    # open first the file with information on 
    # the in-situ features which could be used
    info_dir = settings.get("FEATURES_INFO", None)
    df_sites = pd.read_csv(info_dir)
    site_lst = list(df_sites['siteid'].unique())

    extract_folder = settings.get('EXTRACTION_FOLDER', None)

    # check for which sites preparation could be done
    sites_prep = [item for item in site_lst if item in os.listdir(extract_folder)]
    logger.info(f'FOUND {len(sites_prep)} SITES FOR DATA PREPARATION')

    # Define the temporal aggregation level for which the data should be prepared
    TEMPORAL = settings.get('TEMPORAL_SCALE', None)

    # load the years for which the data should be calibrated
    YEARS = settings.get('YEARS')

    # Load the version of the cal data preparation
    VERSION = settings.get('VERSION')

    overwrite = settings.get('overwrite')

    for temp_scale in TEMPORAL: 
        for year in YEARS:
            basefold_out_plot = basedir.joinpath('data', VERSION,  
                                                  year, 'png', 
                                                  temp_scale)
            basefold_out_plot.mkdir(parents=True, exist_ok=True)
            basefold_out_csv = basedir.joinpath('data', VERSION,
                                                year, 'csv') 
            basefold_out_csv.mkdir(parents=True, exist_ok=True)
             
            for site in sites_prep:
                outname_csv = f'FLUX_EO_cal_data_{site}_{temp_scale}_{year}_{VERSION}.csv'
                if not os.path.exists(basefold_out_csv.joinpath(outname_csv))\
                                or overwrite:
                    df_cal_site = compile_data_calibration(site, settings, 
                                            temp_scale, year)
                    if df_cal_site is None:
                        continue
                    df_cal_site.to_csv(basefold_out_csv.joinpath(outname_csv), 
                                    index=True)
                    
                    # start now the plotting of a few parameters
                    outname_plot = f'FLUX_EO_cal_data_{site}_{VERSION}.png'
                    bands_plot = ['FAPAR', 'CROPSAR', 'GPP_target', 'ssm']
                    plot_cal_TS(bands_plot, df_cal_site,
                                basefold_out_plot.joinpath(outname_plot))
                    
    return