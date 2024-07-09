import openeo
from shapely.geometry import Polygon
connection = openeo.connect('openeo.vito.be').authenticate_oidc()



 # function to return polygon
def bbox(long0, lat0, lat1, long1):
    return Polygon([[long0, lat0],
                    [long1,lat0],
                    [long1,lat1],
                    [long0, lat1]])

geom = bbox(5.661, 52.6457, 52.7335, 5.7298)


SSM = connection.load_collection(
    "CGLS_SSM_V1_EUROPE",
    spatial_extent=None,
    temporal_extent=["2020-05-01",
                     "2020-07-01"]
)

SSM = SSM.resample_spatial(projection=3035, resolution=10.0)

# Composite to dekads
SSM = SSM.aggregate_temporal_period(period="dekad",
                                            reducer="mean")

# Linearly interpolate missing values.
# Shouldn't exist in this dataset but is good practice to do so
SSM = SSM.apply_dimension(dimension="t",
                                process="array_interpolate_linear")




SSM_out = SSM.aggregate_spatial(geom, reducer='mean').filter_temporal().execute()
print('test')





