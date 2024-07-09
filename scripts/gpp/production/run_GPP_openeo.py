""""
This script will be used to run the GPP model on the openEO platform.
The re-calibrated parameter settings will be used to apply and test the model
"""
from pathlib import Path
import os
import json
import openeo
from loguru import logger
from pyproj import Transformer
from cropcarbon.gpp.constants import basefold_prototype
from cropcarbon.utils import get_BBOX
from cropcarbon.utils.permissions import read_write_permission
from cropcarbon.openeo.helper_functions import (test_debug_interpolation,
                                                wrapper_get_datacube_gpp,
                                                wrapper_GPP_est,
                                                test_calc_GDMP_max,
                                                test_calc_GPP)

JOB_OPTIONS = {
    'driver-memory': '5G',
    'driver-memoryOverhead': '2G',
    'driver-cores': '2',
    'executor-memory': '3G',
    'executor-memoryOverhead': '8G',
    'executor-cores': '4',
    'max-executors': '200',
    "udf-dependency-archives": [
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/Evoland/GPP/dep/cropcarbon_0.1.0.zip#tmp/venv_cropcarbon",
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/Evoland/GPP/dep/cropcarbon_dep.zip#tmp/venv_dep"]}


def main_prototype(settings, GPP_config,
                   coords_bbox,
                   connection,
                   buffer=500,
                   mode='debug'):
    # Turn the 2 coordinates used to define the starting
    # point of the BBOX into an actual BBOX
    west, south = coords_bbox.get('WEST'), coords_bbox.get('NORTH') # NOQA
    BBOX = get_BBOX(west, south, buffer=buffer)
    if settings.get('PROJECTION') == 'WGS84':
        # transform the BBOX to WGS84
        transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")
        BBOX_WGS84 = {key: val for key, val in BBOX.items()}
        BBOX_WGS84['south'], BBOX_WGS84['west'] = transformer.transform(BBOX['south'], # NOQA
                                                                        BBOX['west']) # NOQA
        BBOX_WGS84['north'], BBOX_WGS84['east'] = transformer.transform(BBOX['north'], # NOQA
                                                                        BBOX['east']) # NOQA
        BBOX_WGS84['crs'] = f'EPSG:{settings.get("PROJECTION_epsg")}'
        BBOX = BBOX_WGS84
    else:
        BBOX['crs'] = f'EPSG:{settings.get("PROJECTION_epsg")}'
    projection = int(settings.get('PROJECTION_epsg'))
    # get the start and end date for runnning the method
    start_date = settings.get('START_DATE')
    end_date = settings.get('END_DATE')
    
    # check which OpenEO datasets are needed
    # for the GPP model
    datasets = settings.get('DATASETS')
    cube_dataset = None
    
    # check if TEST folder exists
    if not Path(settings.get('TEST_FOLDER')).is_dir():
        os.makedirs(settings.get('TEST_FOLDER'))
        # set permissions for the folder
        read_write_permission(settings.get('TEST_FOLDER'))
    
    if mode == 'input_data':
        cube_dataset = wrapper_get_datacube_gpp(connection, datasets, BBOX,
                                                start_date, end_date)
        # load output format from settings
        output_format = settings.get('OUT_FORMAT')
        outname = f"S2_FAPAR_TEST_CROPSARSCLMASKING_{str(projection)}"
        if output_format == 'NetCDF':
            outname += '.nc'
        elif output_format == 'GTiff':
            outname += '.tiff'  
        # in this case just the input data needed for the UDFs will
        # be downloaded and locally executed
        if Path(os.path.join(settings.get('TEST_FOLDER'),
                             outname)).is_file(): 
            return
        results = cube_dataset.create_job(title="EVOLAND_TEST",
                                          out_format=output_format,
                                          job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
        if output_format == 'NetCDF':
            results.download_file(os.path.join(settings.get('TEST_FOLDER'),
                                               outname))
        elif output_format == 'GTiff':
            results.download_files(os.path.join(settings.get('TEST_FOLDER'),
                                            outname)) # NOQA
        
    elif mode == "local_interpolation_test":
        # test the interpolation UDF locally 
        # load the name of the interpolation technique
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        # load the input data for the interpolation
        input_data = os.path.join(settings.get('TEST_FOLDER'),
                                  f"S2_FAPAR_TEST_CROPSARSCLMASKING_{str(projection)}.nc") # NOQA
        if not Path(input_data).is_file():
            raise ValueError(f'Input data {input_data} not found!')
        # apply the local UDF function
        res = test_debug_interpolation(input_data, interpolation)
        logger.info(f'LOCAL UDF SUCCEEDED WITH OUTCOME: {res}')
        
    elif mode == "run_GDMP_MAX_input_data":
        # Mode in which the UDF for interpolation
        # will be run on the OpenEO platform
        # load the name of the interpolation technique
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        cube_dataset = wrapper_get_datacube_gpp(connection,
                                                datasets,
                                                BBOX,
                                                start_date,
                                                end_date,
                                                projection=projection)
        # check if output file already exists
        if Path(os.path.join(settings.get('TEST_FOLDER'),
                             "GDMP_MAX_INPUT_TEST"
                             f'_{interpolation}'
                             f"_interpolation_{str(projection)}.nc")).is_file(): # NOQA
            return
        results = cube_dataset.create_job(title="EVOLAND_GDMP_MAX_INPUT_TEST",
                                          out_format="NetCDF",
                                          job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
        results.download_file(os.path.join(settings.get('TEST_FOLDER'),
                                           "GDMP_MAX_INPUT_TEST"
                                           f'_{interpolation}'
                                           f"_interpolation_{str(projection)}.nc")) # NOQA
        
    elif mode == 'local_GDMP_max_test':
        # Mode in which the UDF for the GDMP max model
        # will be run on the OpenEO platform
        # load the input data required for GDMP max estimation
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        input_data = Path(os.path.join(settings.get('TEST_FOLDER'),
                                       "GDMP_MAX_INPUT_TEST"
                                       f'_{interpolation}'
                                       f"_interpolation_{str(projection)}.nc"))
        if not Path(input_data).is_file():
            raise ValueError(f'Input data {input_data} not found!')
        # Add meteo scale to config
        GPP_config['METEO_SCALING'] = settings.get('DATASETS').get('AGERA5').get('SCALE') # NOQA
        # apply the local UDF function
        res = test_calc_GDMP_max(input_data, GPP_config)
        logger.info(f'LOCAL UDF SUCCEEDED WITH OUTCOME: {res}')
        
    elif mode == 'run_GDMP_max_test':
        # Mode in which the land cover specific GDMP max
        # estimation will be run on the OpenEO platform
        # load the name of the interpolation technique
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        # check if output file already exists
        if Path(os.path.join(settings.get('TEST_FOLDER'),
                             "GDMP_MAX_TEST"
                             f'_{interpolation}'
                             f"_interpolation_{str(projection)}.nc")).is_file(): # NOQA
            return
        # Add meteo scale to config
        GPP_config['METEO_SCALING'] = settings.get('DATASETS').get('AGERA5').get('SCALE') # NOQA
        cube_GDMP_max = wrapper_GPP_est(connection,
                                        datasets,
                                        BBOX,
                                        start_date,
                                        end_date,
                                        GPP_config,
                                        projection=projection,
                                        GDMP_max_only=True)
     
        results = cube_GDMP_max.create_job(title="EVOLAND_GDMP_MAX_EST_TEST",
                                           out_format="NetCDF",
                                           job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
        results.download_file(os.path.join(settings.get('TEST_FOLDER'),
                                           "GDMP_MAX_TEST"
                                           f'_{interpolation}'
                                           f"_interpolation_{str(projection)}.nc")) # NOQA
    elif mode == 'local_GPP_test':
        # Mode in which the UDF for the GPP model
        # will be run on the OpenEO platform
        # load the input data required for GPP estimation
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        input_data = Path(os.path.join(settings.get('TEST_FOLDER'),
                                       "GDMP_MAX_TEST"
                                       f'_{interpolation}'
                                       f"_interpolation_{str(projection)}.nc")) # NOQA
        if not Path(input_data).is_file():
            raise ValueError(f'Input data {input_data} not found!')
        # apply the local UDF function
        res = test_calc_GPP(input_data, GPP_config)
        logger.info(f'LOCAL UDF SUCCEEDED WITH OUTCOME: {res}')
    elif mode == 'run_GPP_test':
        # Mode in which the land cover specific GPP
        # estimation will be run on the OpenEO platform
        # load the name of the interpolation technique
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        # check from settings output format
        output_format = settings.get('OUT_FORMAT')
        outname = f"GPP_EST_TEST_{interpolation}_interpolation_{str(projection)}"
        if output_format == 'NetCDF':
            outname += '.nc'
        elif output_format == 'GTiff':
            outname += '.tiff'
        # check if output file already exists
        if Path(os.path.join(settings.get('TEST_FOLDER'),
                             outname)).is_file():
            return
        # Add meteo scale to config
        GPP_config['METEO_SCALING'] = settings.get('DATASETS').get('AGERA5').get('SCALE') # NOQA
        cube_GPP = wrapper_GPP_est(connection,
                                   datasets,
                                   BBOX,
                                   start_date,
                                   end_date,
                                   GPP_config,
                                   projection=projection,
                                   GDMP_max_only=False)
        if settings.get('APPLY_SCALING', False):
            # load the scaling 
            scaling = settings.get('SCALING')
            cube_GPP = cube_GPP.linear_scale_range(*scaling)
            
        results = cube_GPP.create_job(title="EVOLAND_GPP_EST_TEST",
                                      out_format=output_format,
                                      job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
        if output_format == 'NetCDF':
            results.download_file(os.path.join(settings.get('TEST_FOLDER'),
                                               outname))
        elif output_format == 'GTiff':
            results.download_files(os.path.join(settings.get('TEST_FOLDER'),
                                                outname))
        
                 
if __name__ == "__main__":
    # Define the version number of the prototype
    version = "V01"
    # mode of running the prototype
    mode = 'run_GPP_test'
    basedir = Path(os.path.join(basefold_prototype, version))
    # open settings file of all data collections
    settingsfile = str(basedir / 'Generic_settings.json')
    settings = json.load(open(settingsfile, 'r'))
    # open config file with parameterization GPP estimation
    configfile = str(basedir / 'GPP_config.json')
    GPP_config = json.load(open(configfile, 'r'))
    # define bounding box for testing
    coords_bbox = settings.get('BBOX')
    # load the size of the buffer for the bounding box around the coordinates
    buffer = settings.get('SIZE', None)
    # load the backend on which the data should be processed
    backend = settings.get('BACKEND')
    # load enpoint connection with Openeo based on platform type
    if backend == "VITO":
        endpoint = openeo.connect("openeo.vito.be").authenticate_oidc()
    elif backend == "cdse":
        endpoint = openeo.connect("openeo.creo.vito.be").authenticate_oidc()
  
    main_prototype(settings,
                   GPP_config,
                   coords_bbox,
                   endpoint,
                   buffer=buffer,
                   mode=mode)
    




