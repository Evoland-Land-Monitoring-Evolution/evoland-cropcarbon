# the year for which the data is released
yr_flx_download = 2023
# the year for which the full year is last available
# and will be used to further complete the dataset
# can be different from the yr_flx_download parameter
# as sometimes not yet a complete year is available
last_full_version_yr = 2022

Version_release = 1


# The CLC class on which there will be a filtered
# for the analysis on crop and grassland
CLC_focus = [211, 212, 213,
             221, 222, 223,
             241, 242,
             231, 321,
             322, 323]


dir_clc = r"/data/cel_vol2/ancillary/LandCover/CLC/clc2018_clc2018_V2018.20b2.tif"  # NOQA
dir_S2_tiles_grid_panEU = r"/data/sigma/Evoland_GPP/data/ancillary/tiling/S2_tiles_panEU.shp"  # NOQA
dir_S2_tiles_grid_world = r"/data/sigma/Evoland_GPP/data/ancillary/tiling/S2_tiles_world.shp"  # NOQA


# the nr of pixels surrounding the site that
# should be used for extraction
window_size_grid = 1


# the basefolder at which the flux data will be stored
basefold_flx = r'/data/sigma/Evoland_GPP/data/ICOS/flux'

# the basefolder where the EO extractions will be stored
basefold_extr = r'/data/sigma/Evoland_GPP/data/ICOS/EO_extractions'

# the basefolder where the S2 grid around the 
# in-situ sites will be stored
basefold_grid = r'/data/sigma/Evoland_GPP/data/ICOS/s2grid'

# the basefolder for the calibration work
basefold_cal = r'/data/sigma/Evoland_GPP/cal'

# the basefolder for the LUE calibration work
basefold_LUE = r'/data/sigma/Evoland_GPP/cal/LUE'

