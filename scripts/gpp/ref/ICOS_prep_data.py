""""
Script that will deal with the different parts of preparing the 
eddy covariance data such that it can be used for model calibration
"""


# import needed packages
import os
from pathlib import Path
import pandas as pd
import numpy as np


from cropcarbon.utils.unpackage import unzip_all
from cropcarbon.gpp.ref.compile_meta_info import (
    update_meta,
    add_ancillary)
from cropcarbon.utils.geom import (
    csvtoshp,
    _create_s2_grid)
from cropcarbon.utils.storage import unique_folder_extension
from cropcarbon.gpp.constants import (
    yr_flx_download,
    Version_release,
    last_full_version_yr,
    dir_clc,
    dir_S2_tiles_grid_panEU,
    CLC_focus,
    window_size_grid,
    basefold_flx,
    basefold_extr,
    basefold_grid
)
from cropcarbon.utils.timeseries import (
    _get_TS_,
    wrapper_TS_agg,
    concat_TS,
    cleaned_TS_agg)
from cropcarbon.utils.plotting import _get_plot_TS_

"""
Import and define first some default meta info about the flux data
"""
overwrite = False
# set to true of the C3/C4 info should be retrieved from the
# previous version
# TODO set to false with the next release
plant_type_prev_year = True


###################################################
# STEP:1 Download most recent data         ########
###################################################

"""
Donwload can be done from this site: https://data.icos-cp.eu/portal/
Use some filter to search the proper data for download. Filters can
should be selected for 'Station class', 'Ecosystem type', 'Data type'

"""

# Unpack all the zipped folders
# Fist unzip the parent folder
folder_dir_donwload = r'/data/sigma/Evoland_GPP/data/ICOS/flux/raw'
folder_name = f'Ecosystem final quality (L2) product in ETC-Archive format - release {str(yr_flx_download)}-{str(Version_release)}'  # NOQA


unzip_all(os.path.join(folder_dir_donwload, folder_name))


###################################################
#         STEP:2 Complete meta info sites #########
###################################################

""""
The meta data summary about the flux sites
will be further completed,
based on the most recent data that became available
"""

# PART1: update the meta CSV-file

# Define the latest csv file with information
# on the data availability from the different
# sites
lst_ref_csv = os.path.join(basefold_flx, 'meta',
                           'flux_site_info_V1_2022.csv')

update_ref_csv = os.path.join(basefold_flx,
                              'meta',
                              f"flux_site_info_V{str(Version_release)}"
                              f"_{str(yr_flx_download)}.csv")

if not os.path.exists(update_ref_csv) or overwrite:
    df_ref_updated = update_meta(os.path.join(folder_dir_donwload,
                                              folder_name),
                                 lst_ref_csv,
                                 last_full_version_yr,
                                 Version_release)

    # save the updated meta data file

    df_ref_updated.to_csv(update_ref_csv, index=False)

else:
    df_ref_updated = pd.read_csv(update_ref_csv)


# PART2: update the georeference file
outshp = os.path.join(Path(update_ref_csv).parent,
                      Path(update_ref_csv).stem + '.shp')
cols = df_ref_updated.columns
if not os.path.exists(outshp) or overwrite:
    csvtoshp(update_ref_csv, lon_hdrname='lon', lat_hdrname='lat',
             fields_to_copy=cols, outshp=outshp)


###################################################
#      STEP:3 Assign ancillary info to sites ######
###################################################
"""
In this section ancillary information will be
added to the meta info of the flux sites like
the corresponding land cover, crop class (C3/C4) 
and S2 tile.
"""
# TODO if new data comes available allow to add the 
# columns of that specific year with information on the crop type
anc_ref_csv = os.path.join(Path(update_ref_csv).parent,
                           Path(update_ref_csv).stem + '_ancillary.csv')
if not os.path.exists(anc_ref_csv) or overwrite:
    flx_meta = add_ancillary(update_ref_csv,
                             dir_clc, 
                             dir_S2_tiles_grid_panEU,
                             CLC_focus, folder_dir_donwload,
                             use_prev_yr_cr_type=plant_type_prev_year)

    flx_meta.to_csv(anc_ref_csv, index=False)

else:
    flx_meta = pd.read_csv(anc_ref_csv)


###################################################
#      STEP:4 Add S2 grid around sites    #########
###################################################

"""
In this section a S2 grid will be created around
the flux sites such that those area can be used
for EO data extraction later on
"""

# Part 1: Create shapefile with the grid for each site

_create_s2_grid(window_size_grid, basefold_grid,
                flx_meta, overwrite=overwrite)

# Part 2: Store the grid for each site into a seperate 
# output folder, which is needed to perform the extraction 
# of EO data
unique_folder_extension(basefold_grid, basefold_extr,
                        overwrite=overwrite, 
                        extension='.shp')


###################################################
#      STEP:5 EC data to CSV per site/year  #######
###################################################

"""
In this section the actual flux data will be
retrieved and stored seperately per year and site
"""

# Part 1: Write TS as CSV

dir_out_dd_TS_csv = os.path.join(basefold_flx,
                                 'csv', 'gpp_dd')
os.makedirs(dir_out_dd_TS_csv, exist_ok=True)

years = np.arange(2016, last_full_version_yr+1)

_get_TS_(flx_meta,
         os.path.join(folder_dir_donwload, folder_name),
         dir_out_dd_TS_csv, years)

# Part 2: Write TS as png

dir_out_dd_TS_png = os.path.join(basefold_flx,
                                 'png', 'gpp_dd')

suffix_plotname = 'GPP_NT_CUT_MEAN_GPP_DT_CUT_MEAN{}_dd.png'
_get_plot_TS_(dir_out_dd_TS_csv,
              dir_out_dd_TS_png,
              suffix_plotname)


###################################################
#      STEP:6 Aggregate and clean TS        #######
###################################################

"""
In this section the daily flux TS will be aggregated.
Different periods for aggregation will be use like
10-daily and monthly.
"""


# Part 1: Aggregate for each site and year the TS to a
# certain defined temporal scale

# overview of the required perios for aggregation and
# the corresponding output folder
dict_periods = {'dekad': os.path.join(basefold_flx,
                                      'csv', 'gpp_dekad'),
                'month': os.path.join(basefold_flx,
                                      'csv', 'gpp_month'),
                'week': os.path.join(basefold_flx,
                                     'csv', 'gpp_week')}

wrapper_TS_agg(dir_out_dd_TS_csv,
               dict_periods)


# Part 2: Clean the daily TS for negative and
#         constant values

concat_TS(dir_out_dd_TS_csv)

# Part 3: Use the cleaned TS to aggregate it
# to the desired period for all sites together

cleaned_TS_agg(os.path.join(basefold_flx,
                            'csv', 'gpp_dd',
                            'grouped'),
               dict_periods)


# Plot also these different aggregations

for period in dict_periods.keys():
    dir_out_period_csv = dict_periods.get(period)
    dir_out_period_png = dir_out_period_csv.replace('csv', 'png')
    suffix_plot_period = 'GPP_NT_CUT_MEAN_GPP_DT_CUT_MEAN{}_' + f'{period}.png'
    os.makedirs(dir_out_period_png, exist_ok=True)
    _get_plot_TS_(dir_out_period_csv,
                  dir_out_period_png,
                  suffix_plot_period)
