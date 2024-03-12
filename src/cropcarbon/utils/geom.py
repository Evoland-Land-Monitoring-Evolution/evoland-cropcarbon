"""
Functions that will deal with geometry objects
"""

import pandas as pd
from shapely.geometry import Point, mapping
from fiona import collection
from rasterio.crs import CRS
import os
import pandas as pd
import fiona
import math
from pyproj import Proj, transform
import fiona
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np


def csvtoshp(incsv, lon_hdrname, lat_hdrname, fields_to_copy, outshp):
    df_csv = pd.read_csv(incsv)

    field_dict = {}
    for field in fields_to_copy:
        field_dict[field] = 'str'
        if field not in df_csv.keys().to_list():
            print(field + ' not in CSV')

    schema = {'geometry': 'Point',
              'properties': field_dict}

    with collection(outshp, "w", "ESRI Shapefile", schema=schema, crs=CRS.from_epsg(4326).wkt
                    ) as output:
        for index, row in df_csv.iterrows():
            point = Point(row[lon_hdrname], row[lat_hdrname])
            properties_dict = {}
            for field in fields_to_copy:
                properties_dict[field] = row[field]

            output.write({'properties': properties_dict,
                         'geometry': mapping(point)})


def _create_s2_grid(windowsize, outdir,
                    metadf, overwrite,
                    dissolve=True):
    """
    Function that will create a Sentinel-2 pixel grid
    around a location.
    If you want to keep the geometry of the separate pixels set 
    dissolve to False. The windowsize defines how many pixels from 
    the center pixels will be taken.
    """

    dx = 10
    dy = 10
    lines = []

    for index, row in metadf.iterrows():
        # check first if a valid CLC class is applicable for the site
        sitelines = []
        print(row['S2_tile'], row['siteid'], row['lat'], row['lon'])
        s2tile = row['S2_tile']

        utm_zone = row['S2_tile'][0:2]
        siteid = row['siteid']
        lat = row['lat']
        lon = row['lon']

        # check if file not already exists:
        outshp_site = os.path.join(outdir, siteid + '_' + s2tile + '.shp')

        if os.path.exists(outshp_site) and not overwrite:
            continue

        in_x = lon
        in_y = lat
        in_epsg = 4326
        out_epsg = int('326'+str(utm_zone))

        inProj = Proj(init=f'epsg:{str(in_epsg)}')
        outProj = Proj(init=f'epsg:{str(out_epsg)}')
        siteutmx, siteutmy = transform(inProj, outProj, lon, lat)

        pixsiteutmx = (int(siteutmx/dx)*dx)  # + tileminx
        windowsiteutmminx = pixsiteutmx - (dx*windowsize)
        windowsiteutmmaxx = pixsiteutmx + (dx*(windowsize+1))

        pixsiteutmy = (int(siteutmy/dx)*dx)
        windowsiteutmminy = pixsiteutmy - (dx*windowsize)
        windowsiteutmmaxy = pixsiteutmy + (dx*(windowsize+1))

        nx = int(math.ceil(abs(windowsiteutmmaxx - windowsiteutmminx)/dx))
        ny = int(math.ceil(abs(windowsiteutmmaxy - windowsiteutmminy)/dy))

        minx = windowsiteutmminx
        maxx = windowsiteutmmaxx
        miny = windowsiteutmminy
        maxy = windowsiteutmmaxy

        xmin = minx
        xmax = maxx
        ymin = miny
        ymax = maxy

        width = dx
        height = dx
        rows = int(np.ceil((ymax - ymin) / height))
        cols = int(np.ceil((xmax - xmin) / width))
        XleftOrigin = xmin
        XrightOrigin = xmin + width
        YtopOrigin = ymax
        YbottomOrigin = ymax - height
        polygons = []
        pixelids = []
        pixelidsv2 = []
        pixelidc = -windowsize

        for i in range(cols):
            Ytop = YtopOrigin
            Ybottom = YbottomOrigin
            pixelidr = windowsize

            for j in range(rows):
                polygons.append(Polygon(
                    [(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
                Ytop = Ytop - height
                Ybottom = Ybottom - height

                pixelids.append("r"+str(pixelidr)+'c'+str(pixelidc))
                pixelidsv2.append("r"+str(j)+'c'+str(i))

                pixelidr = pixelidr - 1

            pixelidc = pixelidc + 1

            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width

        if not dissolve:
            grid = gpd.GeoDataFrame(
                {'pixelidcentre': pixelids, 'pixelidgrid': pixelidsv2, 'geometry': polygons})
        else:
            dissolved_poly = unary_union(polygons)
            pixelid = [siteid]
            grid = gpd.GeoDataFrame(
                {'id': pixelid, 'geometry': dissolved_poly})

        grid.crs = 'epsg:'+str(out_epsg)

        # should be in lat/lon for data extraction
        grid = grid.to_crs(4326)
        grid.to_file(outshp_site)
