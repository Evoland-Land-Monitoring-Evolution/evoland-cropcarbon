""" 
This script will be used to some extractions from
the European Drought Observatory platform. 
"""



# import needed packages
from loguru import logger
import glob
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
import xarray as xr
from pyproj import Transformer
from cropcarbon.cds.extractions import define_batch_jobs
from cropcarbon.utils.timeseries import nc_to_pandas

def sample_data(DATASET, years, data_folder,
                job_id, geom):
    """
    Function that will sample the data out of a file
    """
    # get the corresponding folder 
    # of the dataset fdr sampling
    data_files = glob.glob(os.path.join(data_folder, 
                                        DATASET, '*.nc'))
    
    # define the middle location of the geometry
    lon, lat = geom.centroid.x, geom.centroid.y

    if DATASET in ['SMA', 'CDI']:
        crs_to = 3035
        # need to reproject the data
        transformer = Transformer.from_crs('EPSG:4326',
                                            f'EPSG:{crs_to}')
        lat, lon = transformer.transform(lat, lon)

    # loop now over the years and sample in the corresponding file
    lst_ds_sampled = []
    for yr in years:
        yr_file = [item for item in data_files if str(yr) in Path(item).stem]
        if len(yr_file) == 0:
            raise ValueError(f'No DATASET FOUND FOR {DATASET} ON SITE {job_id}')
        if len(yr_file) > 1:
            raise ValueError(f'Multiple DATASETS FOUND FOR {DATASET} on SITE {job_id}')
        # open the dataset
        ds = xr.open_dataset(yr_file[0])
        # sample based on the geometry
        ds_sampled = ds.sel(lat=lat, lon=lon, method='nearest')
        lst_ds_sampled.append(ds_sampled)
    xr_merged = xr.merge(lst_ds_sampled, compat="no_conflicts")
    return xr_merged


def extract_edo(config, outdir, settings):
    # load the files that contains the information
    data_folder = settings.get("DATA_FOLDER")

    # translation of the collection name to the actual 
    # naming that will be used for the output folder
    translate_col_outfold = {
        "SPI-1_COLLECTION": "SPI-1",
        "SPI-3_COLLECTION": "SPI-3",
        "SMA_COLLECTION": "SMA",
        "CDI_COLLECTION": "CDI"}
    # dictionary that translates the band names to the 
    # actual name they will have in the retrieved data
    dict_translation_VAR = {
        'SPI-1_COLLECTION': {
            'esf01': 'spi-1'},
        'SPI-3_COLLECTION': {
            'esf03': 'spi-3'},
        'SMA_COLLECTION': {
            'smian': 'sma'
        },
         'CDI_COLLECTION': {
            'cdinx': 'cdi'
        }
        }
    # create request based on config file
    for job_id in config.keys():
        site_id = '_'.join(job_id.split('_')[:-1])
        outdir = os.path.join(outdir, site_id, 'extractions')
        os.makedirs(outdir, exist_ok=True)
        period = config.get(job_id).get('PERIOD')
        yrs_range = [int(item[0:4]) for item in period]
        years = [int(item) for item in np.arange(yrs_range[0], yrs_range[-1]+1, 1)]
        GEOM = config.get(job_id).get('GEOM')
        dict_df_collections = {}
        for DATASET in config.get(job_id).get('DATASETS'):
            DATASET_NAME = translate_col_outfold.get(DATASET)
            if DATASET_NAME is None:
                logger.error(f'{DATASET} NOT SUPPORTED')
            ds = sample_data(DATASET_NAME, years, data_folder,
                            job_id, GEOM)
            # rename time axis to ensure consistency
            ds = ds.rename({'time': 't'})
            # convert it now to dataframe format
            band_names =  list(dict_translation_VAR.get(DATASET).keys())
            dict_df_collections = nc_to_pandas(ds, band_names, 
                                               years[0], years[-1], 
                                               dict_df_collections,
                                               translate_col_outfold.get(DATASET), 
                                               clip_year=True,
                                               rename_bands=True, 
                                               translate_bands=dict_translation_VAR.get(DATASET))
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
        extract_edo(config_batch_job, outdir, settings)
    return