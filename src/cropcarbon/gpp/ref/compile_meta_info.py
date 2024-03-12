"""
Compile meta info of flux sites based on most recent data
"""

import os
import glob
import pandas as pd
import rasterio
import numpy as np
from pyproj import Transformer
from pathlib import Path
from loguru import logger as log
from cropcarbon.utils.meta import Satelite_tile_overlap_finder


def update_meta(dir_flx, dir_ref_meta,
                yr_update, Version, 
                start_year = 2016):
    df_ref = pd.read_csv(dir_ref_meta)

    lst_subdirs = ([name for name in os.listdir(dir_flx)
                    if os.path.isdir(os.path.join(dir_flx, name))])

    # years for which the availability has been checked
    # based on previous releases
    years_check = list(np.arange(start_year, 
                                 yr_update,1))

    # add now the release of this year to the check

    years_check.extend([yr_update])

    # initialize the availability of the added year with zero
    df_ref[f'av_{str(yr_update)}'] = [0] * df_ref.shape[0]

    for subdir in lst_subdirs:
        site_id = subdir.split('ICOSETC_')[-1].split('_')[0]
        csv_anc = os.path.join(
            dir_flx, subdir, 'ICOSETC_' + site_id+'_SITEINFO_L2.csv')
        df_anc = pd.read_csv(csv_anc, on_bad_lines='skip')
        lat = df_anc['DATAVALUE'].loc[df_anc['VARIABLE']
                                      == 'LOCATION_LAT'].values[0]
        lon = df_anc['DATAVALUE'].loc[df_anc['VARIABLE']
                                      == 'LOCATION_LONG'].values[0]
        igbp = df_anc['DATAVALUE'].loc[df_anc['VARIABLE'] == 'IGBP'].values[0]

        # check for which years data available

        fn_csv = r'ICOSETC_'+site_id+'_FLUXNET_DD_L2.csv'
        fp_csv = os.path.join(dir_flx, subdir, fn_csv)
        if os.path.isfile(fp_csv) is False:
            continue

        df_flx_site = pd.read_csv(fp_csv, index_col=0, parse_dates=True)

        if (site_id.lower() in df_ref['siteid'].str.lower().values) is False:
            source = f'ICOS{str(yr_update)}Q{str(Version)}'

            # for a new site following columns should be updated

            lst_col_name = []
            lst_av_info = []
            for year in years_check:
                lst_col_name.append(f'av_{str(year)}')
                if df_flx_site.loc[f'{str(year)}-01-01':f'{str(year)}-12-31'].empty:  # NOQA
                    lst_av_info.append(0)
                else:
                    lst_av_info.append(1)

            df_site_info = pd.DataFrame(data=[site_id, lat,
                                              lon, igbp, source,
                                              *lst_av_info]).T
            df_site_info.columns = ['siteid', 'lat',
                                    'lon', 'igbp', 'source',
                                    *lst_col_name]

            df_ref = pd.concat([df_ref, df_site_info])
            df_ref = df_ref.reset_index(drop=True)

        else:
            if df_flx_site.loc[f'{str(yr_update)}-01-01':f'{str(yr_update)}-12-31'].empty:  # NOQA
                av_year = 0
            else:
                av_year = 1

            df_ref.loc[df_ref['siteid'].str.lower() == site_id.lower(),
                       f'av_{str(yr_update)}'] = av_year

    return df_ref


def get_Cr_type_anc(yr, df_anc, site, ICOS_tower=True):
    # translation table to convert the species 
    # C3 or C4 plant type
    dict_translate_type = {
        'Triticum aestivum L.': 'c3',
        'winter wheat': 'c3',
        'Hordeum vulgare L.': 'c3',
        'Vicia faba L.': 'c3',
        'Solanum tuberosum L.': 'c3',
        'Triticum aestivum': 'c3',
        'oats': 'c3',
        'Helianthus Annuus': 'c3',
        'Zea mays L.': 'c4',
        'Phaseolus vulgaris L.': 'c3',
        'Brassica napus L.': 'c3',
        'Beta vulgaris L.': 'c3',
        'winter rapeseed': 'c3',
        'sugarbeet': 'c3',
        'barley': 'c3',
        'Sorghum': 'c4'
    }

    if ICOS_tower:
        # get first the GROUP ID where 
        # a crop type measurement was done at a certain date
        df_filter = df_anc[['VARIABLE', 
                            'DATAVALUE',
                            'GROUP_ID']].astype(str)
        df_date_info = df_filter.loc[df_filter['VARIABLE'] == 'VEG_CHEM_DATE']
        group_info = df_date_info.loc[df_date_info['DATAVALUE'].str.contains(str(yr))]
        if group_info.empty:
            # no crop info available for specific year
            #TODO should raise an error later on once receieved all information
            # raise ValueError(f'NO CROP INFO AVAILABLE FOR {site}')
            return None, None

        # get the group id
        GROUP_ID = group_info['GROUP_ID'].values[0]
        # find now the species related with that group
        SPP_info =  df_filter.loc[((df_filter['GROUP_ID'] == GROUP_ID) & 
                                   (df_filter['VARIABLE'] == 'VEG_CHEM_SPP'))]
        
        if SPP_info.empty:
            raise ValueError(f('NO SPECIES INFORMATION FOR {site}'))
        SPP_info = SPP_info['DATAVALUE'].values[0]
        SPP_C_class = dict_translate_type.get(SPP_info)
    else:
        df_anc = df_anc.astype(str)
        SPP_info = df_anc.loc[df_anc.year == yr]['crop type'].values[0]
        SPP_C_class = df_anc.loc[df_anc.year == yr]['class'].values[0]
    return SPP_info, SPP_C_class
    


def add_crop_class(df, basedir, metadir, year, version,
                   use_prev_yr=False,
                   start_year = '2017'):
    """
    Function that will add to each retained site,
    information on the actual type of crop that was
    cultivated and if it belongs to a C3/C4 variant.

    The parameter 'use_prev_yr' is added to cope with 
    the problem that this information was absent in 
    the first release of 2023. This parameter should 
    be set to false for later updates. 
    """
    if use_prev_yr:
        year = str(int(year)-1)
        if version[1] == 2:
            version = 'V1'
        else:
            version = 'V2'
    folder_raw = f'Ecosystem final quality (L2) product in ETC-Archive format - release {str(year)}-{str(version[-1])}' #NOQA

    # get now the folder in which need to be searched
    folder_flx_raw = os.path.join(basedir, folder_raw)

    # for each site seperately define the corresponding crop type
    sites = df.siteid.unique().tolist()

    # years for which the crop type should be 
    # determined based on previous releases
    years_check = list(np.arange(int(start_year), 
                                 int(year)+1,1))
    years_check = [str(item) for item in years_check]

    col_yrs = [item for item in df.columns if item.split('_')[0] == 'av']

    lst_df_site_update_meta = []
    for site in sites: 
        df_site = df.loc[df.siteid == site]
        # check years for which we need to find type
        yrs_av = ['_'.join(col_yrs[i].split('_')[1:]) for i in range(len(col_yrs)) if (df_site[col_yrs[i]] == 1).all()]

        # check first if it is not a grassland, as this is a C3 crop (in EU)
        igbp = df_site['igbp'].values[0]
        if igbp != 'GRA':
            # get the ancillary file of the site
            file_anc = glob.glob(os.path.join(folder_flx_raw,
                                        f'ICOSETC_{site}_*', 
                                        f'ICOSETC_{site}_ANCILLARY_L2.csv'))
            if len(file_anc) == 0:
                log.info('NO CROP TYPE CLASS INFORMATION AVAILABLE'
                        ' --> SEARCH IN NON-ICOS')
                file_anc = glob.glob(os.path.join(Path(metadir).parent,
                                                'C3_C4_info', 'Non_ICOS',
                                                f'{site}.csv'))
                # not an ICOS tower (other data format)
                ICOS_tower = False
                if len( file_anc) == 0:
                    raise ValueError(f'NO CROP TYPE INFO FOR SITE {site}')
                df_anc = pd.read_csv(file_anc[0], sep=';')
            else:
                ICOS_tower = True
                try:
                    df_anc = pd.read_csv(file_anc[0])
                except:
                    df_anc = pd.read_csv(file_anc[0], on_bad_lines='skip',
                                         sep=';')


        for yr in years_check:
            # check if this information is already provided
            if f'C_{yr}' in df_site.columns:
                continue
            if yr not in yrs_av:
                df_site.loc[:, (f'C_{yr}')] = None
                df_site.loc[:, (f'Cr_{yr}')] = None
                continue
            if not igbp == 'GRA':
                Cr , C_class = get_Cr_type_anc(yr, df_anc, site, 
                                         ICOS_tower=ICOS_tower)
                df_site.loc[:, (f'C_{str(yr)}')] = C_class
                df_site.loc[:, (f'Cr_{str(yr)}')] = Cr
            else:
                df_site.loc[:, (f'C_{str(yr)}')] = 'c3' 
                df_site.loc[:, (f'Cr_{str(yr)}')] = 'grassland'
        lst_df_site_update_meta.append(df_site.reset_index(drop=True))
    df_crop_info = pd.concat(lst_df_site_update_meta)
    df_crop_info = df_crop_info.reset_index(drop=True)
    return df_crop_info
        


def add_ancillary(dir_flx, dir_clc,
                  dir_S2_tiles, CLC_focus, 
                  dir_raw, use_prev_yr_cr_type=False,
                  igb_focus = ['CRO', 'GRA']):

    # open the meta file
    flx_meta = pd.read_csv(dir_flx)

    # keep only certain IGBP classes
    flx_meta = flx_meta.loc[flx_meta['igbp'].isin(igb_focus)]

    lst_CLC_DN = []
    lst_S2tile_name = []
    for i, site in flx_meta.iterrows():
        lat, lon = site.lat, site.lon

        # Load the CLC info
        with rasterio.open(dir_clc) as src:
            crs_to = src.crs
            crs_to = crs_to.to_string().rsplit("AUTHORITY")[-1][-7:-3]
            # convert to correct projection system first
            transformer = Transformer.from_crs('EPSG:4326',
                                               f'EPSG:{crs_to}')
            lat_reproj, lon_reproj = transformer.transform(lat, lon)

            samplegenerator_rad = src.sample([(lon_reproj, lat_reproj)])

            value_lst_rad = [x[0] for x in list(samplegenerator_rad)]
            if value_lst_rad:
                value_lst_rad = int(value_lst_rad[0])
            else:
                value_lst_rad = None

            lst_CLC_DN.append(value_lst_rad)

        # Load the S2 tile info
        tile_info = Satelite_tile_overlap_finder([lat, lon],
                                                 data_type='POINT',
                                                 tiling_dir=dir_S2_tiles)
        # if multiple tiles just take the first one
        if len(tile_info) > 0:
            tile_ID = tile_info[0]
        else:
            tile_ID = None

        lst_S2tile_name.append(tile_ID)

    flx_meta['CLC_DN'] = lst_CLC_DN
    flx_meta['S2_tile'] = lst_S2tile_name
    # keep only certain CLC sites
    #flx_meta = flx_meta.loc[flx_meta.CLC_DN.isin(CLC_focus)]

    # keep only sites with S2 tiles in PAN-EU
    flx_meta = flx_meta[flx_meta['S2_tile'].notna()]

    # add C3/C4 info to sites
    # get version and year info of the dataset
    Version = Path(dir_flx).stem.split('_')[-2]
    Year = Path(dir_flx).stem.split('_')[-1]
    flx_meta = add_crop_class(flx_meta, dir_raw, dir_flx, 
                                Year, Version, 
                                use_prev_yr=use_prev_yr_cr_type)

    return flx_meta
