import os
from osgeo import gdal
from cropcarbon.utils.permissions import read_write_permission


def _add_scale_factor(tiff_file: str, outfile: str,
                      scale_factor: float = 0.001) -> None:
    """
    Function that will add the scale factor to the raster file
    """
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=YES -a_scale {scale_factor} {tiff_file} {outfile}" # NOQA
    os.system(cmd)
    read_write_permission(outfile)
    
    
def _to_cog(tiff_file, outfile):
    """
    Function that will convert the raster file to COG
    """
    cmd = f"gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=1024 -co RESAMPLING=BILINEAR -co OVERVIEWS=IGNORE_EXISTING {tiff_file} {outfile}" # NOQA
    os.system(cmd)
    read_write_permission(outfile)
    

def _add_color_table(tiff_file, color_table, band_description=None):
    """
    Function that will add the color table to the raster file
    """
    ds = gdal.Open(tiff_file, 1)
    band = ds.GetRasterBand(1)
    # define the band name of the raster
    if band_description is not None:
        band.SetDescription(band_description)
    # create color table
    colors = gdal.ColorTable()
    # open the color table file and read the content
    with open(color_table, 'r') as f:
        content = f.read()

    # set color for each value
    lst_new_lines = []
    for line in content.split('\n'):
        if line:
            color = line.split(',')
            lst_new_lines.append(color)
            # repeat the color for all values between 20-65534
            if int(color[0]) == 20:
                for i in range(21, 65534):
                    color_line = [i, color[1], color[2],
                                  color[3], color[4], i]
                    lst_new_lines.append(color_line)
    for color in lst_new_lines:
        colors.SetColorEntry(int(color[0]), (int(color[1]), int(color[2]),
                                             int(color[3]), int(color[4])))    
    # set color table and color interpretation
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    # close and save file
    del band, ds       
    pass


def _add_band_description(tiff_file, band_description):
    """
    Function that will add the band description to the raster file
    """
    # use gdal to open the raster file
    ds = gdal.Open(tiff_file, 1)
    # get the first band
    band = ds.GetRasterBand(1)
    # set the band description
    band.SetDescription(band_description)
    # close and save the file
    del band, ds
    pass
