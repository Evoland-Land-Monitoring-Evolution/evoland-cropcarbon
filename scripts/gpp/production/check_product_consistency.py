"""
Within this script the aim is to define some AOI's spread across 
the test sites needed to check the consistency of the product.
Afterwards the consistency check will be done   
"""

import os
import utm
import json
import shutil
import math
import openeo
import numpy as np
import rasterio
import pandas as pd
import pyproj
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform
from cropcarbon.openeo.helper_functions import wrapper_GPP_est
from cropcarbon.gpp.constants import basefold_prototype
from cropcarbon.utils.permissions import read_write_permission
from glob import glob


# function that will filter on the folder
# needed for LC check
def find_folder(geom, LC_folder) -> str:
    lat_centroid, lon_centroid = geom.centroid.y, geom.centroid.x
    # remove the decimals from the coordinates
    lat_centroid = math.trunc(lat_centroid)
    lon_centroid = math.trunc(lon_centroid)
    # load list of LC folders
    lst_folders = os.listdir(LC_folder)
    # find the folder that match with the centroid
    for folder in lst_folders:
        # focus only on second last _ folder
        folder_substr = folder.split('_')[-2]
        if lat_centroid > 0:
            lat_orient = 'N'
        else:
            lat_orient = 'S'
        if lon_centroid > 0:
            lon_orient = 'E'
        else:
            lon_orient = 'W'
        if not folder_substr.startswith(lat_orient) or not lon_orient in folder_substr: # NOQA
            continue
        if (int(folder_substr[1:3])+3 > lat_centroid and int(folder_substr[1:3]) <= lat_centroid): # NOQA
            # now do the same check for the longitude
            if (int(folder_substr[4:7])+3 > lon_centroid and int(folder_substr[4:7]) <= lon_centroid): # NOQA
                return folder
            

# now create function that will enable to check if the geom is covered enough with the LC of interest
def check_LC(geom, LC_folder, LC_VALUES=[30, 40],
             perc_cover=.8) -> bool:
    # open the raster only on the window where the geom is located
    # check all files in folder
    LC_files = glob(os.path.join(LC_folder, '*_Map.tif'))
    with rasterio.open(LC_files[0]) as src:
        window = rasterio.windows.from_bounds(*geom.bounds, src.transform)
        arr = src.read(window=window, boundless=True)
    # check if at least perc cover of the area is
    # covered with the LC of interest
    # do the count of occurences of LC VALUES
    count = np.isin(arr, LC_VALUES).sum()
    # check if this at least covers perc_cover of the area
    if count/arr.size > perc_cover:
        return True
    else:
        return False
    
      
# write function that sample from shapefile object
# random squares with certain kernel size
def sample_AOI(shp_AOI, LC_folder,
               kernel_size=128, n_sample_per_AOI=6,
               pixel_size=10):
    """ sample the AOI's from the shapefile object"""
    # get the unique locations
    lst_gdf = []
    shp_AOI['id'] = shp_AOI['location'] + '_' + shp_AOI['Name']
    lst_locations = list(shp_AOI['id'].unique())
    # now sample the AOI's with n sample per AOI
    for loc in lst_locations:
        loc_rename = loc.replace('-', '').replace(' ', '')
        # get the AOI's for this location
        shp_AOI_loc = shp_AOI_sample[shp_AOI_sample['id'] == loc]
        # sample the AOI's
        for idx, row in shp_AOI_loc.iterrows():
            # get the geometry
            geom = row['geometry']
            # get the centroid
            centroid = geom.centroid
            utm_zone = utm.from_latlon(centroid.y, centroid.x)
            # based on UTM zone define the EPSG
            if utm_zone[2] > 0:
                epsg = 32600 + utm_zone[2]
            else:
                epsg = 32700 + abs(utm_zone[2])
            # use pyproj to transform the geom to the UTM zone
            wgs84 = pyproj.CRS("EPSG:4326")
            utm_proj = pyproj.CRS(f"EPSG:{epsg}")
            project = pyproj.Transformer.from_crs(wgs84, utm_proj,
                                                  always_xy=True).transform
            geom_UTM = transform(project, geom)   
            # get the bounding box
            minx, miny, maxx, maxy = geom_UTM.bounds
            # get the kernel size
            kernel = kernel_size
            # define area with kernel size that is fully located in the AOI
            # sample the AOI's
            np.random.seed(42)
            counter = 0
            while counter < n_sample_per_AOI:
                # sample the location and ensure seeding for reproducibility
                x = np.random.uniform(minx+(kernel*pixel_size),
                                      maxx-(kernel*pixel_size))
                y = np.random.uniform(miny+(kernel*pixel_size),
                                      maxy-(kernel*pixel_size))
                # create the square
                square = box(x, y, x+(kernel*pixel_size),
                             y+(kernel*pixel_size))
                # transform back to WGS84
                project_back = pyproj.Transformer.from_crs(utm_proj, wgs84,
                                                           always_xy=True).transform # NOQA
                square = transform(project_back, square)
                # find the correct worldcover file that can be used to check
                # if the square
                # is located in grassland cropland area
                name_folder = find_folder(square, LC_folder)
                if name_folder is None:
                    continue
                
                res_check = check_LC(square, os.path.join(LC_folder,
                                                          name_folder))
                if not res_check:
                    continue
                
                # store the square in a geopandas object
                gdf = gpd.GeoDataFrame(geometry=[square])
                # set column with id
                gdf['id'] = f'{loc_rename}_{str(counter)}'
                lst_gdf.append(gdf)
                counter += 1
    # concat the list of geodataframes
    gdf = gpd.GeoDataFrame(pd.concat(lst_gdf, ignore_index=True))
    gdf.crs = shp_AOI.crs
    return gdf


# below a function used to retrieve the CGLOPS GPP 300 m product
def get_datacube_GPP300(connection, BBOX, start_date, end_date,
                        projection=3035, collection='CGLS_GDMP300_V1_GLOBAL',
                        RT_version="RT6"):
    # load the datacube
    GPP300 = connection.load_collection(collection,
                                        spatial_extent=BBOX,
                                        temporal_extent=[start_date,
                                                         end_date],
                                        properties={
                                            "productGroupId":
                                                lambda x: x == RT_version
                                        },
                                        bands=['GDMP'])
    if projection == 3035:
        GPP300 = GPP300.resample_spatial(projection=projection,
                                         resolution=10)
    GPP300 = GPP300.rename_labels('bands', ['GDMP300'])
    GPP300 = GPP300.filter_bbox(BBOX)
    return GPP300
    
                  
if __name__ == "__main__":
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

    # define some filters and settings
    ids = ['3', '4', '6']
    Version = 'V01'
    years = ['2018', '2021']
    kernel_size = 100
    product = 'GPP10M'
    # load settingsfile for GPP estimation
    settingsfile = os.path.join(basefold_prototype, Version,
                                'Generic_settings.json')
    configfile = os.path.join(basefold_prototype, Version,
                              'GPP_config.json')
    # open the files
    with open(settingsfile, 'r') as file:
        settings = json.load(file)
    with open(configfile, 'r') as file:
        config = json.load(file)
    # add to config meteo scaling info
    config['METEO_SCALING'] = settings.get('DATASETS').get('AGERA5').get('SCALE') # NOQA
    # Load the folder at which the AOI's are stored
    basefold_AOIs = os.path.join(basefold_prototype, 'test_sites',
                                 'Evoland_prototype_sites_v3_implementation_AGRI.gpkg') # NOQA
    # define basefold of worldcover
    basefold_worldcover = r'/data/MTDA/WORLDCOVER/ESA_WORLDCOVER_10M_2020_V100/MAP' # NOQA
    # check if actually is a file
    if not os.path.exists(basefold_AOIs):
        raise ValueError('The AOI file does not exist')
    outfold = os.path.join(basefold_prototype, Version,
                           'samples_consistency')
    outname = 'AOI_samples_consistency.gpkg'
    # check if the folder exists
    if not os.path.exists(outfold):
        os.makedirs(outfold)
        read_write_permission(outfold)
        shutil.chown(outfold, user='bontek', group='vito')
   
    """
    PART THAT CREATES THE SAMPLES
    """

    # check if the samples are not yet generated
    if not os.path.exists(os.path.join(outfold, outname)):
        # load the AOI's
        shp_AOI = gpd.read_file(basefold_AOIs)
        lst_locations = list(shp_AOI['location'].unique())
        # now only retain those location that start with the right number
        lst_locations = [loc for loc in lst_locations
                         if loc.startswith(tuple(ids))]
        # filter on the relevant AOI's
        shp_AOI_sample = shp_AOI[shp_AOI['location'].isin(lst_locations)]
        
        # do the sampling of random location for checks
        gdf_samples = sample_AOI(shp_AOI_sample, basefold_worldcover,
                                 kernel_size=kernel_size)
        gdf_samples.to_file(os.path.join(outfold, outname), driver='GPKG')
        read_write_permission(outfold)
        
    """
    PART THAT WILL DO THE EXTRACTIONS OF THE SAMPLES
    """
    connection = openeo.connect("openeo.vito.be").authenticate_oidc()
    
    # load the samples file
    gdf_samples = gpd.read_file(os.path.join(outfold, outname))
    # convert to json like format
    gdf_samples_json = json.loads(gdf_samples.to_json())
    
    # load the datasets that are required in OpenEO
    datasets = settings.get('DATASETS')
    
    # now extract the samples
    for year in years:
        extract_folder_year = os.path.join(outfold, year)
        if not os.path.exists(extract_folder_year):
            os.makedirs(extract_folder_year)
            read_write_permission(extract_folder_year)
            shutil.chown(extract_folder_year, user='bontek', group='vito')
    
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        if product == 'GPP300M':
            # group the iteration based on the tile ID
            tiles_unique = list(set([item.rsplit('_')[1]
                                    for item in gdf_samples['id']]))
        else:
            tiles_unique = gdf_samples['id'].unique()
        
        # loop over the tiles
        # for idx, row in gdf_samples.iterrows():
        for tile in tiles_unique:
            # filter the gdf on the tile ID
            gdf_samples_tile = gdf_samples[gdf_samples['id'].str.contains(tile)]
            # get the geometry
            bbox = gdf_samples.total_bounds
            # set BBOX in dictionary format with west, east, north, south keys
            bbox = {'west': bbox[0], 'east': bbox[2],
                    'north': bbox[3], 'south': bbox[1],
                    'crs': 'EPSG:4326'}
            # load the name of the field
            field_name = tile 
            pol_aggregate = json.loads(gdf_samples_tile.to_json())
            # create the output folder
            if product == 'GPP10M':
                outname_sample = f'{product}_{start_date}_{end_date}_{field_name}_whittaker.csv' # NOQA
                # check if not yet exists
                if os.path.exists(os.path.join(extract_folder_year,
                                               outname_sample)):
                    continue
                cube_GPP = wrapper_GPP_est(connection, datasets, bbox,
                                           start_date, end_date,
                                           config, projection=3035)
                
                cube_GPP = cube_GPP.aggregate_spatial(pol_aggregate,
                                                      reducer='mean')
                results = cube_GPP.create_job(title=f'{product}_{field_name}_{year}', # NOQA
                                              out_format='CSV',
                                              job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
            elif product == 'GPP300M':
                outname_sample = f'{product}_{start_date}_{end_date}_{field_name}_RT6.csv' # NOQA
                # check if not yet exists
                if os.path.exists(os.path.join(extract_folder_year,
                                               outname_sample)):
                    continue
                if int(year) < 2021:
                    RT_version = "RT5"
                else:
                    RT_version = "RT6"
                cube_GDMP = get_datacube_GPP300(connection, bbox,
                                                start_date, end_date,
                                                projection=4326,
                                                RT_version=RT_version)
                cube_GDMP = cube_GDMP.aggregate_spatial(pol_aggregate,
                                                        reducer='mean')
                results = cube_GDMP.create_job(title=f'{product}_{field_name}_{year}', # NOQA
                                               out_format='CSV',
                                               job_options=JOB_OPTIONS).start_and_wait().get_results() # NOQA
                   
            results.download_file(os.path.join(extract_folder_year,
                                               outname_sample))
            shutil.chown(os.path.join(extract_folder_year,
                                      outname_sample),
                         user='bontek', group='vito')
            if product == 'GPP10M':
                outfold_per_field = os.path.join(extract_folder_year,
                                                 f'{product}_per_field_whittaker') # NOQA
            else:
                outfold_per_field = os.path.join(extract_folder_year,
                                                 f'{product}_per_field_RT6')
            if not os.path.exists(outfold_per_field):
                os.makedirs(outfold_per_field)
                read_write_permission(outfold_per_field)
                shutil.chown(outfold_per_field, user='bontek',
                             group='vito')
            # now open the result and rename the feature
            # index column to the proper name
            df_result = pd.read_csv(os.path.join(extract_folder_year,
                                                 outname_sample))
            for feature_index in df_result['feature_index'].unique():
                # get the name of the field that is associated
                # with the field index
                field_id = gdf_samples_tile.iloc[feature_index]['id']
                df_result.loc[df_result['feature_index'] == feature_index,
                              'feature_index'] = field_id
            # rename the last column to GPP 300 M
            df_result.rename(columns={df_result.columns.values[-1]: product,
                                      'feature_index': 'id'},
                             inplace=True)
            # save the dataframe for each id separate
            for field_id in df_result['id'].unique():
                df_result_id = df_result[df_result['id'] == field_id]
                df_result_id = df_result_id.sort_values(by='date')
                if product == 'GPP300M':
                    # apply scale factor
                    df_result_id[product] = df_result_id[product] * 0.02
                    # convert now GDMP to GPP
                    df_result_id[product] = df_result_id[product] * 0.05
                outname_sample = f'{product}_{start_date}_{end_date}_{field_id}.csv' # NOQA
                # save result
                df_result_id.to_csv(os.path.join(outfold_per_field,
                                                 outname_sample),
                                    index=False)
                shutil.chown(os.path.join(outfold_per_field,
                                          outname_sample),
                             user='bontek', group='vito')