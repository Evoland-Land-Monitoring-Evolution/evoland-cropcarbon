"""
Script that will be used to add color table to the raster files
and define the applied scaling factor used in the product
"""

import os
from pathlib import Path
from glob import glob
from cropcarbon.utils.permissions import read_write_permission
from cropcarbon.openeo.postprocessing.utils import (_add_scale_factor,
                                                    _add_color_table)


if __name__ == "__main__":
    # define folder where raster data is stored
    RASTER_FOLDER = r"/data/sigma/Evoland_GPP/prototype/V01/test_region/cost_est/AOI_2x2_31UFS" # NOQA
    # load all tiff files in folder
    tiff_files = glob(os.path.join(RASTER_FOLDER, '**', '*.tif'))
    # load color table
    COLOR_TABLE = r"/data/sigma/Evoland_GPP/prototype/V01/test_region/GPP_10M_color_table.txt" # NOQA
    
    # define OUTPUT folder
    OUT_FOLDER = Path(RASTER_FOLDER).parent.joinpath(Path(RASTER_FOLDER).name + '_postproc') # NOQA
    # create folder if not exists
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
        # check if we have read and write permission
        read_write_permission(OUT_FOLDER)
    
    # loop over the tiff files and apply adding scale factor and color table
    for tiff_file in tiff_files:
        # first apply scale factor
        _add_scale_factor(tiff_file,
                          OUT_FOLDER.joinpath(Path(tiff_file).name))
        # now apply color table
        _add_color_table(OUT_FOLDER.joinpath(Path(tiff_file).name).as_posix(),
                         COLOR_TABLE) # NOQA
        
        
    



