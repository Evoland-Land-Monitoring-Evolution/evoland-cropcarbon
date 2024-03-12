"""
Code to do extractions from the copernicus climate store
"""

# import needed packages
from loguru import logger
import cdsapi
import glob
import geopandas as gpd
import numpy as np
import os
import cfgrib
import xarray as xr
from cropcarbon.openeo.extractions import (
     split_batch_jobs,
     check_availabilty)
from cropcarbon.utils.timeseries import nc_to_pandas


def retrieve_request(c, DATASET, years, outdir, config, 
                     job_id, outname_tmp, area_reformat):
    if DATASET == 'EVAPORATION_COLLECTION':
            dataset_name_eds = 'reanalysis-era5-land'
    else:
        raise AttributeError(f'EXTRACT FROM {DATASET}'
                                    'not yet supported')

    bands = config.get(job_id).get('DATASETS').get(DATASET)\
            .get('BANDS')
    
    # list of the months to adress for each split
    dict_months_split = {
        0: ['01', '02', '03',
            '04', '05', '06'],
        1: ['07', '08', '09', 
            '10', '11', '12']}
    # store all the requested data locations in a list:
    lst_loc_data = []

    # split up the request in max 6 months data period
    iteration = 0
    for year in years:
        # year should be splitted in two
        for split in range(2):
            months = dict_months_split.get(split)
            outname_request = outname_tmp.split('.grib')[0] + \
                              f'_{str(iteration)}.grib'
            c.retrieve(
                    dataset_name_eds,
                    {
                        'day': ['01', '02', '03', '04', '05',
                                '06', '07', '08', '09', '10',
                                '11', '12', '13', '14', '15',
                                '16', '17', '18', '19', '20',
                                '21', '22', '23', '24', '25',
                                '26', '27', '28', '29', '30',
                                '31'],
                        'month': months,
                        'year': [str(year)],
                        'time': [
                            '00:00', '01:00', '02:00', '03:00',
                            '04:00', '05:00', '06:00', '07:00',
                            '08:00', '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00', '15:00',
                            '16:00', '17:00', '18:00', '19:00',
                            '20:00', '21:00', '22:00', '23:00'
                        ],
                        'variable': bands,
                        'area': area_reformat,
                        'format': 'grib'
                    },
                    os.path.join(outdir, outname_request))
            iteration += 1 
            lst_loc_data.append(os.path.join(outdir, outname_request))
    # merge all the files and concatenate them
    lst_ds = [cfgrib.open_dataset(item) for item in lst_loc_data]
    xr_merged = xr.merge(lst_ds, compat="no_conflicts")
    return xr_merged



def extract_cds(config, outdir):
    # connect the cds backend
    c = cdsapi.Client()
    # dictionary that translates the band names to the 
    # actual name they will have in the retrieved data
    dict_translation_VAR = {
        "EVAPORATION_COLLECTION": {
            'potential_evaporation': 'pev',
            'total_evaporation': 'e'}
    }
    # translation of the collection name to the actual 
    # naming that will be used for the output folder
    translate_col_outfold = {
        "EVAPORATION_COLLECTION": 'ET'
    }

    # create request based on config file
    for job_id in config.keys():
        site_id = '_'.join(job_id.split('_')[:-1])
        outdir = os.path.join(outdir, site_id, 'extractions')
        outname_tmp = f'{job_id}_out_cds.grib'
        os.makedirs(outdir, exist_ok=True)
        period = config.get(job_id).get('PERIOD')
        yrs_range = [int(item[0:4]) for item in period]
        years = [int(item) for item in np.arange(yrs_range[0], yrs_range[-1]+1, 1)]
        GEOM = config.get(job_id).get('GEOM')
        area = list(GEOM.bounds)
        area_reformat = [area[1], area[0], area[-1], area[2]]
        dict_df_collections = {}
        for DATASET in config.get(job_id).get('DATASETS'):
            ds = retrieve_request(c, DATASET, years, outdir,
                                             config, job_id, outname_tmp, 
                                             area_reformat)
            # take already the mean within a certain day
            ds = ds.groupby('time.date').mean().mean(dim='step')
            # take the mean across time if multiple pixels
            ds = ds.stack(pos=("longitude", "latitude"))
            # rename time axis to ensure consistency
            ds = ds.rename({'date': 't'})
            # convert it now to dataframe format
            band_names = [dict_translation_VAR.get(DATASET).get(key) 
                          for key in dict_translation_VAR.get(DATASET).keys()]  
            dict_df_collections = nc_to_pandas(ds, band_names, 
                                               years[0], years[-1], 
                                               dict_df_collections,
                                               translate_col_outfold.get(DATASET), 
                                               clip_year=True)
            # remove now all files related with the original extraction
            files_remove = glob.glob(os.path.join(outdir, f'{outname_tmp.split(".")[0]}_*'))
            [os.unlink(item) for item in files_remove]
        for dataset in dict_df_collections.keys():
            collection_name = '_'.join(dataset.split('_')[:-1])
            year_dataset = dataset.split('_')[-1]
            outfold = os.path.join(outdir, collection_name, year_dataset)
            os.makedirs(outfold, exist_ok=True)
            outname = f'{site_id}_{collection_name}_{str(year_dataset)}.csv'
            dict_df_collections.get(dataset).to_csv(os.path.join(outfold, 
                                                    outname),
                                                    index=True)
            
    logger.info('Final dataframes saved!')


def define_batch_jobs(outdir, dataset, settings):
    logger.info(f'Preparing {dataset} for extractions...')
    datasetdir = outdir / dataset
    extractionsdir = outdir / dataset / 'extractions'
    # get the sample files
    # get start and end date
    start_date = settings.get('START', None)
    end_date = settings.get('END', None)
    # now define for which periods the extractions 
    # should be done. If for example only for certain 
    # years in the time range in-situ data is available
    # the batch jobs will be subdivided in consecutive
    # years for which data is available. 
    batch_job_ids, periods = split_batch_jobs(settings, start_date, end_date,
                                              dataset)
    # Also important is to assess for each period if 
    # the data is not yet available. 
    # If not, for which datasets in the period 
    # the processing should be still done.
    # if for a part of the period the processing is already done, 
    # the period should be made smaller
    periods, collection_info_periods, batch_job_ids = check_availabilty(settings, periods,
                                                                        batch_job_ids ,
                                                                        extractionsdir.as_posix())
    if not periods:
        logger.info(f'EXTRACTION FOR {dataset} ALREADY DONE --> SKIP')
        return {}
    # get the geometry information for extraction
    samplefiles = glob.glob(str(datasetdir / f'{dataset}_*.shp'))
    if len(samplefiles) == 0:
        raise ValueError(f'NO SHAPE INFORMATION AVAILABLE FOR {dataset}')
    if len(samplefiles) > 1:
        raise ValueError(f'TOO MANY SHAPE INFORMATION AVAILABLE FOR {dataset}')
    geom = gpd.read_file(samplefiles[0]).geometry.values[0]

    # now create the config file for the extraction request
    dict_config = {}

    for i in range(len(batch_job_ids)):
        jobid = batch_job_ids[i]
        col_name = list(collection_info_periods.keys())[i]
        collections_extract = collection_info_periods.get(col_name)
        dict_col_extract_info = {}  
        for col in collections_extract:
            # check if the corresponding col should be still extracted
            if not collection_info_periods.get(col_name).get(col):
                # collection already extracted
                continue
            bands = settings.get('DATASETS').get(col).get('BANDS')
            dict_col_extract_info.update({col: {
                                        'NAME': col,
                                        'BANDS': bands}})
        # get now info on the bands that should be 
        dict_config.update({jobid: {'PERIOD': periods[i],
                                    'GEOM': geom,
                                    'DATASETS': dict_col_extract_info}})
    return dict_config



def main(outdir, datasets, settings):
    logger.info(f'GETTING ALL INPUTS FOR EXTRACTIONS')

    for dataset in datasets:
        # define batch jobs for extractions on cds
        config_batch_job = define_batch_jobs(outdir, 
                                             dataset, 
                                             settings)
        if not config_batch_job:
            continue
        logger.info(f'STARTING PROCESSING FOR {dataset}')
        extract_cds(config_batch_job, outdir)
    return