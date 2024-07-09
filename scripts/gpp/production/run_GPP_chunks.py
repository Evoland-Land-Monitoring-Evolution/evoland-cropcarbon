"""
Code that will be used to run the GPP prototype for
a certain BBOX, subdvided in equal chunks to reduce
the processing time.    
"""

from pathlib import Path
import os
import json
import openeo
from loguru import logger
from pyproj import Transformer
from glob import glob
from cropcarbon.gpp.constants import basefold_prototype
from cropcarbon.utils import get_BBOX
from cropcarbon.utils.permissions import read_write_permission
from cropcarbon.openeo.helper_functions import wrapper_GPP_est
from cropcarbon.openeo.postprocessing.utils import (_add_scale_factor,
                                                    _add_color_table,
                                                    _add_band_description,
                                                    _to_cog)

JOB_OPTIONS = {
    'driver-memory': '5G',
    'driver-memoryOverhead': '2G',
    'driver-cores': '2',
    'executor-memory': '6G',
    'executor-memoryOverhead': '8G',
    'executor-cores': '4',
    'max-executors': '200',
    "udf-dependency-archives": [
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/Evoland/GPP/dep/cropcarbon_0.1.0.zip#tmp/venv_cropcarbon",
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/Evoland/GPP/dep/cropcarbon_dep.zip#tmp/venv_dep"]}


def mosaic_rasters(in_folder: str, tile_name: str = None):
    # load all the tiff files of the same date and mosaic them
    files = glob(os.path.join(in_folder, "GPP_fuseTS_32631_chunk_*", "*.tif"))
    outfolder = os.path.join(in_folder, f"GPP_{tile_name}")
    # check if file already exists
    if not Path(outfolder).is_dir():
        os.makedirs(outfolder)
        # set permissions for the folder
        read_write_permission(outfolder)
    # store all the tiff files with the same name into a dictionary with the
    # name and as value all the files with the same name
    dict_files = {}
    for file in files:
        name = Path(file).stem
        if name in dict_files:
            dict_files[name].append(file)
        else:
            dict_files[name] = [file]
    for period in dict_files.keys():
        # load the date from the period key
        date = period.split("_")[-1]
        outname = f"GPP_31UFS_{date}.tif"
        # load the files that should be merged
        files = dict_files[period]
        files = " ".join(files)
        if os.path.exists(os.path.join(outfolder, outname)):
            logger.info(f"Files for date {date} are already mosaiced")
            continue
        # merge and ensure nodata is still nodata
        cmd = f"gdal_merge.py -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=YES -o {os.path.join(outfolder, outname)} -n 65535 -a_nodata 65535 {files}" # NOQA
        
        os.system(cmd)
    

def split_chunks(BBOX: dict, CHUNK_SIZE: int) -> list:
    """
    Code that will split the BBOX in equal chunks

    Args:
        BBOX (dict): corner coordinates of BBOX
        CHUNK_SIZE (int): size of the new created chunks
    """
    # get the coordinates of the BBOX
    west, south = BBOX.get('west'), BBOX.get('south')
    east, north = BBOX.get('east'), BBOX.get('north')
    # get the size of the BBOX
    size_x = east - west
    size_y = north - south
    # calculate the number of chunks in x and y direction
    chunks_x = int(size_x / CHUNK_SIZE)
    chunks_y = int(size_y / CHUNK_SIZE)
    # calculate the new size of the chunks
    new_size_x = size_x / chunks_x
    new_size_y = size_y / chunks_y
    # create the new chunks
    chunks = []
    for i in range(chunks_x):
        for j in range(chunks_y):
            chunk = {
                'west': west + i * new_size_x,
                'south': south + j * new_size_y,
                'east': west + (i + 1) * new_size_x,
                'north': south + (j + 1) * new_size_y,
                'crs': BBOX.get('crs')
                
            }
            chunks.append(chunk)
    return chunks


def run_chunks(settings, GPP_config,
               coords_bbox, connection, CHUNK_SIZE,
               CHUNK_SELECTION='all'):
    
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
    # check if TEST folder exists
    if not Path(settings.get('TEST_FOLDER')).is_dir():
        os.makedirs(settings.get('TEST_FOLDER'))
        # set permissions for the folder
        read_write_permission(settings.get('TEST_FOLDER'))
    # now split up the BBOX in equal chunks
    lst_chunks = split_chunks(BBOX, CHUNK_SIZE)
    if CHUNK_SELECTION != 'all':
        lst_chunks = lst_chunks[CHUNK_SELECTION[0]:CHUNK_SELECTION[1]]
        logger.info(f"Processing chunks {CHUNK_SELECTION[0]} to {CHUNK_SELECTION[1]}") # NOQA
        
    # loop per chunk and request the output
    for chunk in lst_chunks:
        logger.info(f"Processing chunk {lst_chunks.index(chunk)}"
                    f" OUT OF {len(lst_chunks)}")
        chunk_name = f"{int(chunk.get('west'))}_{int(chunk.get('south'))}_{int(chunk.get('east'))}_{int(chunk.get('north'))}" # NOQA
        
        # load the name of the interpolation technique
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        # check from settings output format
        output_format = settings.get('OUT_FORMAT')
        outname = f"GPP_{interpolation}_{str(projection)}_chunk_{chunk_name}"
        if output_format == 'NetCDF':
            outname += '.nc'
        elif output_format == 'GTiff':
            outname += '.tiff'
        # create output folder based on coordinates
        output_folder = os.path.join(settings.get('TEST_FOLDER'), 
                                     outname) # NOQA
        # check if output folder exists
        if not Path(output_folder).is_dir():
            os.makedirs(output_folder)
            # set permissions for the folder
            read_write_permission(output_folder)
        else:
            # check if not yet already tiff files in folder
            files = glob(os.path.join(output_folder, "*.tif"))
            if files:
                logger.info(f"Chunk {chunk} is already processed")
                continue
        # Add meteo scale to config
        GPP_config['METEO_SCALING'] = settings.get('DATASETS').get('AGERA5').get('SCALE') # NOQA
        cube_GPP = wrapper_GPP_est(connection,
                                   datasets,
                                   chunk,
                                   start_date,
                                   end_date,
                                   GPP_config,
                                   projection=projection,
                                   GDMP_max_only=False)
        if settings.get('APPLY_SCALING', False):
            # load the scaling 
            scaling = settings.get('SCALING')
            cube_GPP = cube_GPP.linear_scale_range(*scaling)
            
        results = cube_GPP.create_job(title=f"EVOLAND_chunk_{chunk_name}",
                                      out_format=output_format,
                                      job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
        if output_format == 'NetCDF':
            results.download_file(os.path.join(settings.get('TEST_FOLDER'),
                                               outname))
        elif output_format == 'GTiff':
            results.download_files(os.path.join(settings.get('TEST_FOLDER'),
                                                outname))
        logger.info(f"Chunk {chunk} is processed")


if __name__ == "__main__": 
    # Define the version number of the prototype
    version = "V01"
    tile = '31UFS'
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
    # CHUNK SIZE OF JOBS
    CHUNK_SIZE = 10980
    # subdivide the processing to 4 parellel running terminals
    CHUNK_SELECTION = 'all'
    # Define the mode that the script should be executed
    mode = ['OPENEO_RUNNING', 'POSTPROCESSING'] 
    # Location of selected color table for adding to raster
    color_table = r"/data/sigma/Evoland_GPP/prototype/V01/test_region/GPP_10M_color_table.txt" # NOQA
    # below the naming convention for the output will be defined
    prefix_name = 'C5_CropGrasslandGPP'
    # define version of product
    Version = 'v1.0'

    # load enpoint connection with Openeo based on platform type
    if backend == "VITO":
        endpoint = openeo.connect("openeo.vito.be").authenticate_oidc()
    elif backend == "cdse":
        endpoint = openeo.connect("openeo.creo.vito.be").authenticate_oidc()
    
    if 'OPENEO_RUNNING' in mode:    
        # run the chunks
        run_chunks(settings, GPP_config, coords_bbox, endpoint, CHUNK_SIZE,
                   CHUNK_SELECTION)

    if 'POSTPROCESSING' in mode:
        # Mosiac files into tile 
        mosaic_rasters(settings.get('TEST_FOLDER'), tile_name=tile)
        # apply postprocessing routines to the raster in the mosaic folder
        path_mosaic_folder = os.path.join(settings.get('TEST_FOLDER'), f'GPP_{tile}') # NOQA
        files_mosaic = glob(os.path.join(path_mosaic_folder, "*.tif")) # NOQA
        # define OUTPUT folder
        path_postproc = path_mosaic_folder + '_postproc' # NOQA
        # create folder if not exists
        if not os.path.exists(path_postproc):
            os.makedirs(path_postproc)
            # check if we have read and write permission
            read_write_permission(path_postproc)
        for file_mosaic in files_mosaic:
            # get the date from the outfile
            date = Path(file_mosaic).stem.split('_')[-1][0:10].replace('-', '')
            # define the outname of the product
            # based on the fixed naming convention
            outname_product = prefix_name + f'_{tile}_' + date + f'_GPP_{Version}.tif' # NOQA
            outfile = os.path.join(path_postproc, outname_product)
            if os.path.exists(outfile.replace('.tif', '_cog.tif')):
                logger.info(f"File {file_mosaic} is already postprocessed")
                continue
            _add_scale_factor(file_mosaic,
                              os.path.join(path_postproc,
                                           outname_product))
            # add band description to the raster
            _add_band_description(outfile,
                                  band_description='GPP [gC/m2/day]')
            # apply color table to the raster
            _add_color_table(outfile, color_table)
            # transform to COG
            _to_cog(outfile, outfile.replace('.tif', '_cog.tif'))
            os.unlink(outfile)
        