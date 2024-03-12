import shapefile
from shapely.geometry import Point  # Point class
# shape() is a function to convert geo objects through the interface
from shapely.geometry import shape
from shapely.geometry import Polygon


def get_tile_overlap(geom, data_type, tiling_shp_dir,
                     tile_boundary_test=False):

    if data_type == 'POINT':
        lat = geom[0]
        lon = geom[1]
    tile_insitu = []
    tiles_boundary = []
    s2_tiling = tiling_shp_dir
    shp = shapefile.Reader(s2_tiling)
    all_shapes = shp.shapes()
    all_records = shp.records()

    if data_type == 'POINT':
        point = (lon, lat)
    tiles_point = []
    point_tile_boundary = []
    for i in range(len(all_shapes)):
        boundary = all_shapes[i]  # get a boundary polygon
        if data_type == 'POINT':
            # make a point and see if it's in the polygon
            if Point(point).within(shape(boundary)):
                # get the second field of the corresponding record
                name = all_records[i][0]
                tiles_point.append(name)
        else:
            if Polygon(geom.within(shape(boundary))):
                name = all_records[i][0]
                tiles_point.append(name)
            if tile_boundary_test:
                if shape(boundary).intersects(Polygon(geom)) and not Polygon(geom).within(shape(boundary)):
                    point_tile_boundary.append(all_records[i][0])
        if i == len(all_shapes)-1:
            # tiles_point = set(tiles_point)
            # use accolades in case point in tile overlap
            tile_insitu = tiles_point
            if tile_boundary_test and not point_tile_boundary:
                tiles_boundary.append(['0'])
            if tile_boundary_test and point_tile_boundary:
                tiles_boundary.append(point_tile_boundary)

    if tile_boundary_test:
        return tile_insitu, tiles_boundary

    else:
        return tile_insitu


def Satelite_tile_overlap_finder(geom, data_type,
                                 tiling_dir, tile_boundary_test=False):

    if tile_boundary_test:
        tile_insitu, tile_boundary = get_tile_overlap(geom, data_type, tiling_dir,
                                                      tile_boundary_test=tile_boundary_test)
        result = (tile_insitu, tile_boundary)
    else:
        tile_insitu = get_tile_overlap(geom, data_type, tiling_dir,
                                       tile_boundary_test=tile_boundary_test)
        result = tile_insitu
    return result
