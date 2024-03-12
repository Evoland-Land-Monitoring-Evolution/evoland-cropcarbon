"""
Extractions functions to request some data from OpenEO
"""


# import needed packages

from loguru import logger
import glob
from pathlib import Path
import geopandas as gpd
import json
import pandas as pd
from datetime import datetime
import fire
import xarray as xr
import h3.api.basic_int as h3
import numpy as np
import os
from shutil import copy2
from datetime import datetime
from dateutil.relativedelta import relativedelta


from openeo.extra.job_management import MultiBackendJobManager


from cropcarbon.openeo.connection import openeo_prod
from cropcarbon.openeo.preprocessing import gpp_preprocessed_inputs
from cropcarbon.openeo.config.settings import get_job_options
from cropcarbon.utils.timeseries import (
    nc_to_pandas, 
    json_to_pandas
)



def get_gpp_features(connection, provider, gdf, start, end,
                           settings):

    # Select the right collections depending on provider
    
    s2col = settings.get('S2_COLLECTION', None)

    if s2col is not None and s2col:
        # not use the actual terrascope layer as
        #  it could be that not all data is available there
        if provider == 'terrascope':
            s2col = 'SENTINEL2_L2A' #'TERRASCOPE_S2_TOC_V2'
        elif provider.isin(['sentinelhub', 'creodias']):
            s2col = 'SENTINEL2_L2A'
        else:
            raise ValueError(f'Provider {provider} not supported!')

    cropsarcol = settings.get('CROPSAR_COLLECTION', None)
    if cropsarcol is not None and cropsarcol:
        cropsarcol = 'CROPSAR'

    meteocol = settings.get('METEO_COLLECTION', None)
    if meteocol is not None and meteocol:
        meteocol = 'AGERA5'
    ssmcol = settings.get('SSM_COLLECTION', None)
    if ssmcol is not None and ssmcol:
        ssmcol = 'CGLS_SSM_V1_EUROPE'

    processing_options = settings.get('processing_options')

    # geometry determined by sample points
    bbox = None

    # geometry for aggregation
    geo = json.loads(gdf.to_json())

    # call actual function computing the inputs
    datacube = gpp_preprocessed_inputs(
        connection, bbox, start, end,
        S2_collection=s2col,
        METEO_collection=meteocol,
        SSM_collection=ssmcol,
        CROPSAR_collection=cropsarcol,
        preprocess=False,
        geo=geo,
        **processing_options)

    if not cropsarcol:
        # now only retain the data for the points of interest
        return datacube.aggregate_spatial(geo, reducer='mean')
    else:
        # aggregation is already done
        return datacube



def extraction_function(row, connection_provider, connection,
                        provider):

    # get settings for the row
    start = row['start']
    end = row['end']

    # add buffer to period 
    # to allow interpolation methods at tails
    start = datetime.strptime(start, "%Y-%m-%d") - relativedelta(months=+6)
    start = start.strftime("%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d") + relativedelta(months=+6)
    end = end.strftime("%Y-%m-%d")


    dataset_name = row['ref_id']
    fnp = row['FILENAME']
    filename = str(Path(fnp).name)
    settingsfile = row['settingsfile']
    settings = json.load(open(settingsfile, 'r'))
    collection_keys = [x for x in row.index if 'COLLECTION' in x]
    for col in collection_keys:
        settings.update({col: bool(int(row[col]))})

    # Read samples
    pols = gpd.read_file(fnp)
    
    # get cropclass features for these points
    features = get_gpp_features(connection, provider,
                                      pols, start, end, settings)

    job_options = get_job_options(task='extractions',
                                  provider=provider)

    job = features.create_job(
        title=f"GPP_CalibFeat_{dataset_name}_{provider}",
        description=f"GPP calibration extractions. Source of samples: {filename}",
        out_format="NetCDF",
        job_options=job_options)

    print(job)

    return job

def extract_samples(dataframe, status_file,
                    provider="terrascope",
                    parallel_jobs=1):

    manager = CustomJobManager()
    c = openeo_prod
    

    manager.add_backend(provider, connection=c,
                        parallel_jobs=parallel_jobs)

    manager.run_jobs(
        df=dataframe,
        start_job=extraction_function,
        output_file=Path(status_file)
    )


def run_extractions(status_file,
                    dataframe,
                    provider="terrascope"):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    extract_samples(dataframe, status_file,
                    provider=provider, parallel_jobs=1)

def split_batch_jobs(settings, start, end,
                     siteid, 
                     keyid='siteid'):
    # Define the amount of needed batch jobs
    # and the corresponding time range per 
    # batch job
    df_av = pd.read_csv(settings.get('ANCILLARY_FOLDER', None))

    # Years for which the extraction should be done
    years = np.arange(int(start[0:4]), int(end[0:4])+1, 1)
    years = [str(x) for x in years]

    # Check if for all years data should be processed
    df_av_site = df_av.loc[df_av[keyid] == siteid]
    
    # Availability check
    av_info = [df_av_site[f'av_{year}'].values[0] for year in years]
    av_info = [item == 1 for item in av_info]

    # check if CropSAR collection 
    # should be extracted
    # If so, duplication of jobs is needed
    cropsar = settings.get('CROPSAR_COLLECTION', None)

    # Check if for full period available
    if all(av_info):
        batch_job_ids = [f'{siteid}_1']
        periods = [[start, end]]
    else:
        indices_True = [i for i, x in enumerate(av_info) if x == True]

        # consecutive period of availabilty
        if np.all(np.diff(indices_True) == 1):
            batch_job_ids = [f'{siteid}_1']
            start = f'{years[indices_True[0]]}-01-01'
            end = f'{years[indices_True[-1]]}-12-31'
            periods = [[start, end]]

        # multiple splits of the period needed
        else:
            loc_split = np.where(np.diff(indices_True) > 1)[0]

            batch_job_ids = []
            periods = []
            
            for i in range(len(loc_split)+1):
                if i == 0:
                    start = f'{years[indices_True[0]]}-01-01'
                    end = f'{years[indices_True[loc_split[i]]]}-12-31'

                elif i == 1:
                    start = f'{years[indices_True[i+1]]}-01-01'
                    end = f'{years[indices_True[-1]]}-12-31'
                else:
                    raise Exception('More than two splits of the dataset not yet supported!')
                
                batch_job_ids.append(f'{siteid}_{str(i+1)}')
                periods.append([start, end])
    # duplicate ids and periods if cropsar
    if cropsar:
        cropsar_ids = [item + '_cs' for item in batch_job_ids]
        batch_job_ids.extend(cropsar_ids)
        periods.extend(periods)

    return batch_job_ids, periods


def check_availabilty(settings, periods, job_ids, outdir):
    translate_col_outfold = {
        'METEO_COLLECTION': 'AGERA5',
        'SSM_COLLECTION': 'S1_SSM',
        'S2_COLLECTION': 'S2_FAPAR',
        'CROPSAR_COLLECTION': 'CROPSAR',
        "EVAPORATION_COLLECTION": 'ET',
        "SPI-1_COLLECTION": "SPI-1",
        "SPI-3_COLLECTION": "SPI-3",
        "SMA_COLLECTION": "SMA",
        "CDI_COLLECTION": "CDI"
    }
    
    # check which collections should be retrieved
    collection_keys = [item for item in settings.keys() if 'COLLECTION' in item]
    # create now dict with only the collections that should be extracted
    collections_extract = [col for col in collection_keys if settings.get(col)]

    # for each period and each year in 
    # the period check if the data is 
    # already available
    lst_periods = []

    # keep info on the collections 
    # to be processed for the period
    dict_collections_period = {}

    # list that will save the batch job 
    # ids that should be kept
    lst_job_ids_final = []

    for i in range(len(periods)):
        period = periods[i]
        name_job = job_ids[i]
        years = np.arange(int(period[0][0:4]), 
                          int(period[-1][0:4])+1, 1)
        
        lst_years_keep = []
        # collections that should 
        # be still extracted
        lst_col_to_extract = []
        for year in years:
            # list with info if collection 
            # has already been extracted
            lst_col_extracted = []

            for col in collections_extract:
                if col not in translate_col_outfold.keys():
                    logger.error(f'{col} NOT SUPPORTED')

                # skip check if not job dedicated to CropSAR
                if 'CROPSAR' in col and not 'cs' == name_job.split('_')[-1]: 
                    continue
                if 'cs' == name_job.split('_')[-1] and not 'CROPSAR' in col: 
                    continue
                # get outfold
                outfold_col_yr = os.path.join(outdir,
                                translate_col_outfold.get(col), 
                                str(year))
                # check if already data present 
                # for this collection
                files = glob.glob(os.path.join(outfold_col_yr, '*.csv'))
                if len(files) > 0:
                    lst_col_extracted.append(True)
                else:
                    lst_col_extracted.append(False)
                    lst_col_to_extract.append(col)
            if any(lst_col_to_extract):
                lst_years_keep.append(year)
        
        # now check for the (updated) period which 
        # collections should be retained
        if lst_years_keep:
            period_new = [f'{str(lst_years_keep[0])}-01-01',
                          f'{str(lst_years_keep[-1])}-12-31']
            lst_periods.append(period_new)
            collection_extract_info = [True if x in lst_col_to_extract else False for x in collection_keys]
            collection_info_extract_period = dict(zip(collection_keys,collection_extract_info))
            dict_collections_period.update({f'collections_{str(i+1)}': collection_info_extract_period})
            lst_job_ids_final.append(name_job)
    

    return lst_periods, dict_collections_period, lst_job_ids_final
                

    
def json_batch_jobs(basedir, dataset, settingsfile):

    logger.info(f'Preparing {dataset} for openeo extractions...')

    datasetdir = basedir / dataset
    extractionsdir = basedir / dataset / 'extractions'
    jsons_dir = extractionsdir / 'jsons'
    jsons_dir.mkdir(exist_ok=True, parents=True)

    # get the sample files
    samplefiles = glob.glob(str(datasetdir / f'{dataset}_*.shp'))

    # open settings
    settings = json.load(open(settingsfile, 'r'))

    # split the ones which haven't been split yet
    for samplefile in samplefiles:
        sname = Path(samplefile).stem
        # split_file = jsons_dir / f'{sname}_split_overview.json'
        # if not split_file.exists():

        logger.info(f'Splitting samplefile {sname} '
                    'into manageable chuncks for openeo...')

        # get start and end date
        start_date = settings.get('START', None)
        end_date = settings.get('END', None)
        logger.info('Date range for extractions: '
                    f'{start_date} - {end_date}')

        # now define for which periods the extractions 
        # should be done. If for example only for certain 
        # years in the time range in-situ data is available
        # the batch jobs will be subdivided in consecutive
        # years for which data is available. 
        site_name = '_'.join(sname.split('_')[0:-1])
        batch_job_ids, periods = split_batch_jobs(settings, start_date, 
                                                    end_date, site_name)

        # Also important is to assess for each period if 
        # the data is not yet available. 
        # If not, for which datasets in the period 
        # the processing should be still done.
        # if for a part of the period the processing is already done, 
        # the period should be made smaller
        # CropSAR extraction will be treathed separately
        periods, collection_info_periods, batch_job_ids = check_availabilty(settings, periods,
                                                                            batch_job_ids ,
                                                                            extractionsdir)

        # in case no processing needed anymore, 
        # just skip and remove file if present
        if not periods:
            outname =  f"{sname}_split_overview.json"
            if Path(os.path.join(jsons_dir, outname)).is_file():
                os.rename(os.path.join(jsons_dir, outname), 
                os.path.join(jsons_dir, outname.split('.json')[0] +'_finished.json'))
            return

        add_attributes = {'settingsfile': settingsfile}
        add_attributes.update(collection_info_periods)
        _json_batch_jobs(samplefile, jsons_dir, sname,
                        periods, batch_job_ids,
                        add_attributes=add_attributes)

        logger.info(f'{sname} split successfully!')

    return


def _json_batch_jobs(infile, json_dir, source_name, periods,
                   batch_jobs, id_attr='id',
                   add_attributes=None):

    logger.info('Reading shapefile...')
    df = gpd.read_file(infile)

    logger.info('Preparing geometry...')
    cellsh3 = df.geometry.centroid.apply(lambda x: h3.geo_to_h3(x.y, x.x, 10))
    df['s2cell'] = cellsh3
    df.index = cellsh3
    df.sort_index(inplace=True)

    logger.info('Only retaining crucial attributes...')
    df = df[[id_attr, 's2cell', 'geometry']]

    logger.info('Splitting up dataset...')

    ref_id = '_'.join(source_name.split('_')[0:-1])

    if not batch_jobs: 
        return

    if len(batch_jobs) > 1:
        filenames = []
        for name in batch_jobs :
            f = json_dir / f"{ref_id}_group_{name}.json"
            df['batch_id'] = [name]
            df['start'] = [periods[batch_jobs.index(name)][0]]
            df['end'] = [periods[batch_jobs.index(name)][1]]
            filenames.append(str(f))
            df.to_file(f)
        
        union = df.geometry.aggregate(lambda s: s.unary_union)
        polys = pd.Series(data=np.repeat([union.convex_hull], len(batch_jobs)),
                            index=np.arange(0,len(batch_jobs),1), name="GEOM")    
        count = pd.Series(data=np.repeat([df.shape[0]], len(batch_jobs)), 
                          index=np.arange(0,len(batch_jobs),1),
                          dtype=np.int64, name="COUNT") 
        logger.info(f'Dataset to be split into {len(filenames)} parts!')

    else:
        logger.info('No splitting required for this one!')
        count = pd.Series(data=[len(df)], index=[0],
                          dtype=np.int64, name="COUNT")
        name = 'all'
        filenames = [str(json_dir / f"{source_name}_group_{name}.json")]
        df['batch_id'] = [batch_jobs[0]]
        df['start'] = [periods[0][0]]
        df['end'] = [periods[0][1]]
        df.to_file(filenames[0])
    
        union = df.geometry.aggregate(lambda s: s.unary_union)
        polys = pd.Series(data=[union.convex_hull],
                            index=[0], name="GEOM")    
    
    logger.info('Creating splits dataframe...')
    splits_frame = gpd.GeoDataFrame({"COUNT": count, "FILENAME": pd.Series(
        filenames)}, geometry=polys.reset_index().GEOM)

    logger.info('Adding some attributes...')
    # if no splitting period, no subdivision 
    # of extraction period should be made per frame
    if splits_frame.shape[0] == 1:
        splits_frame['ref_id'] = [batch_jobs[0]] * len(splits_frame)
        splits_frame['start'] = [periods[0][0]] * len(splits_frame)
        splits_frame['end'] = [periods[0][1]] * len(splits_frame)
        if add_attributes is not None:
            for title, value in add_attributes.items():
                if type(value) == dict:
                    for title2, value2 in value.items():
                         splits_frame[title2] = [value2] * len(splits_frame)     
                else:
                    splits_frame[title] = [value] * len(splits_frame)
              


    else:
        lst_df = []
        for i, row in splits_frame.iterrows():
            row['ref_id'] = [batch_jobs[i]][0]
            row['start'] = [periods[i][0]][0]
            row['end'] = [periods[i][1]][0]
            if add_attributes is not None:
                for title, value in add_attributes.items():
                    if not 'collection' in title:
                        row[title] = value
                    else:
                        idx_collection = int(title.split('_')[-1])
                        if idx_collection != i+1:
                            continue
                        title = title.split('_')[0]
                        if type(value) == dict:
                            for title2, value2 in value.items():
                                row[title2] = value2    
                        else:
                            row[title] = value  

            lst_df.append(pd.DataFrame(row).T)
        splits_frame = gpd.GeoDataFrame(pd.concat(lst_df, ignore_index=True),
                                        geometry = splits_frame.geometry)
  
    logger.info('Saving split dataframe...')
    splits_frame.to_file(
        str(json_dir / f"{source_name}_split_overview.json"), index=False)
    logger.info('Split dataframe saved!')


class CustomJobManager(MultiBackendJobManager):

    
                

    def on_job_done(self, job, row):

        fnp = row['FILENAME']
        target_dir = Path(fnp).parents[1]
        job_metadata = job.describe_job()
        target_dir = target_dir / job_metadata['title']
        target_dir.mkdir(exist_ok=True, parents=True)
        job.get_results().download_files(target=target_dir)

        # define output folder
        csv_dir = target_dir.parent

        with open(target_dir / f'job_{job.job_id}.json', 'w') as f:
            json.dump(job_metadata, f, ensure_ascii=False)

        # copy geometry to result directory
        try:
            copy2(fnp, target_dir)
        except:
            print(f'COPY ERROR {fnp} {target_dir}')
        
        dict_df_collections = {}

        # open json containing the geometries 
        # and meta info
        polys = gpd.read_file(fnp)
        
        if bool(int(row['CROPSAR_COLLECTION'])):
            # open extractions json
            infile = glob.glob(str(target_dir / 'result.json'))
            if len(infile) == 0:
                logger.warning('No extractions found!')
                return
            with open(infile[0], 'r') as out:
                data = json.load(out)

            # CropSAR TS extraction
            if 'CROPSAR' in list(data.keys())[0]:
                cropsar_bands = ['FAPAR']
                year_start = polys['start'][0][0:4]
                year_end = polys['end'][0][0:4]
                dict_df_collections = json_to_pandas(data, cropsar_bands, 
                                                    year_start, year_end, 
                                                    dict_df_collections, 
                                                    'CROPSAR')


        else:
            # open extractions netcdf
            infile = glob.glob(str(target_dir / 'timeseries.nc'))
            if len(infile) == 0:
                logger.warning('No extractions found!')
                return
            data = xr.open_dataset(infile[0])

            # do check on completeness
            if data.feature.shape[0] != len(polys):
                logger.warning('Extractions incomplete, ignoring these!')
                os.rename(infile[0],
                        infile[0].replace('.nc', '_INCOMPLETE.nc'))
                return

            # add ref_id
            refID = [row['ref_id']] * len(polys)
            data = data.assign_coords(refID=("feature", refID))

            # add years_processed
            year_start = [x[0:4] for x in polys['start'][0:4]] * len(polys)
            data = data.assign_coords(year_start=("feature", year_start))

            year_end = [x[0:4] for x in polys['end'][0:4]] * len(polys)
            data = data.assign_coords(year_end=("feature", year_end))

            # remove feature_names coordinate
            data = data.reset_coords(names=['feature_names'],
                                    drop=True)

            # save netcdf
            outfile = infile[0].replace('.nc', '_fin.nc')
            data.to_netcdf(path=outfile)
            logger.info('Final netcdf saved!')

            # reformat data to store per year 
            # and per dataset everything in a 
            # separate csv file
            logger.info('Start conversion to pandas format')

            # METEO DATA FROM AGERA5
            if 'temperature_max' in data.variables.keys():
                # extract the corresponding bands
                meteo_bands = ['temperature_max', 
                               'temperature_min',
                               'temperature_mean',
                               'solar_radiation_flux',
                               'temperature_dewpoint',
                               'vapour_pressure']
                dict_df_collections = nc_to_pandas(data, meteo_bands, 
                                                    year_start[0], year_end[0], 
                                                    dict_df_collections, 
                                                    'AGERA5')
            # FAPAR DATA FROM S2
            if 'FAPAR' in data.variables.keys():
                # extract the corresponding bands
                fapar_bands = ['FAPAR']
                dict_df_collections = nc_to_pandas(data, fapar_bands, 
                                                    year_start[0], year_end[0], 
                                                    dict_df_collections, 
                                                    'S2_FAPAR')

            # SSM DATA FROM S1
            if 'ssm' in data.variables.keys():
                # extract the corresponding bands
                ssm_bands = ['ssm']
                dict_df_collections = nc_to_pandas(data, ssm_bands, 
                                                    year_start[0], year_end[0], 
                                                    dict_df_collections, 
                                                    'S1_SSM')

        id_site = '_'.join(row['ref_id'].split('_')[:-1])
        for dataset in dict_df_collections.keys():
            collection_name = '_'.join(dataset.split('_')[:-1])
            year_dataset = dataset.split('_')[-1]
            outfold = csv_dir.joinpath(collection_name, year_dataset)
            outfold.mkdir(parents=True, exist_ok=True)
            outname = f'{id_site}_{collection_name}_{str(year_dataset)}.csv'
            dict_df_collections.get(dataset).to_csv(os.path.join(outfold, 
                                                    outname),
                                                    index=True)

        logger.info('Final dataframes saved!')



def main(basedir, provider):

    logger.info(f'GETTING ALL INPUTS FOR EXTRACTIONS')

    # set output dir
    status_dir = basedir / 'extraction_status'
    status_dir.mkdir(exist_ok=True, parents=True)

    # screen directory of jsons to check for datasets to be added
    # to extractions
    dfs = glob.glob(str(basedir / '*' / 'extractions' / 'jsons' /
                        '*_split_overview.json'))

    # if extractions were already started previously, skip these
    check_files = [Path(x).parent / f'{Path(x).stem}_dummy.json'
                   for x in dfs]
    check = [x.is_file() for x in check_files]
    to_add = []
    check_file_to_create = []
    for i, df in enumerate(dfs):
        if not check[i]:
            to_add.append(df)
            check_file_to_create.append(check_files[i])
        else:
            # if the dummy file is empty (due to problem launching)
            # remove it and start processing
            f = open(check_files[i])
            # returns JSON object as
            # a dictionary
            info = json.load(f)

            if not bool(info):
                # dummy file is empty so processing can be done
                to_add.append(df)
                check_file_to_create.append(check_files[i])
                os.unlink(check_files[i])


    n_datasets = len(to_add)
    if n_datasets > 0:

        logger.info(f'Found {len(to_add)} datasets for extractions')
        logger.info('Merging datasets for this batch...')
        dfs = []
        for df_to_add in to_add:
            dfs.append(gpd.read_file(df_to_add))
        df = pd.concat(dfs, axis=0, ignore_index=True)

        # create dummy files for these datasets so they do not end up
        # in next round of extractions
        for outfile in check_file_to_create:
            data = {}
            with open(outfile, 'w') as fp:
                json.dump(data, fp)

        # save dataframe
        time = datetime.now().strftime('%Y%m%d-%H%M%S')
        df_file = str(status_dir / f'{time}_df.json')
        df.to_file(df_file, format='GeoJSON')

        logger.info('Datasets added!')

    logger.info('STARTING ACTUAL EXTRACTIONS...')
    ref_id_new = []
    df_files = glob.glob(str(status_dir / '*_df.json'))
    for df_file in df_files:
        df = gpd.read_file(df_file)
        status_file = df_file.replace('df.json', 'status.csv')
        # check if all extractions have been done or not
        launch = True
        if Path(status_file).is_file():
            # TODO add here some code that will check 
            # if for the requested datasets all the data is there or not
            status_df = pd.read_csv(status_file, header=0)
            status_df_todo = status_df.loc[status_df["status"] != 'finished']
            if len(status_df_todo) == 0:
                launch = False
            else:
                ref_id_new.extend(status_df_todo.ref_id.unique().tolist())

        if launch:
            logger.info(f'LAUNCHING {len(df)} JOBS FROM {df_file}!')
            fire.Fire(run_extractions(status_file, df, provider))

            # TODO: when all extractions are done, the script should stop
            # and move on, which is not the case if there are extractions
            # remaining with status "error"

    logger.info('Merging extractions per ref_id...')
    # remove duplicates in list of ref_ids to check
    ref_id_new = list(dict.fromkeys(ref_id_new))

    logger.success('All done!')
    return