## in this script some function that will be often used by the services will be added
from pathlib import Path

import openeo
from openeo.internal.graph_building import PGNode
from openeo.rest.datacube import DataCube
from openeo.processes import date_shift, array_element, eq

def load_udf(udf):
    with open(udf, 'r+', encoding="utf8") as f:
        return f.read()

#Write content to named file
def fwrite(fname, fcontent):
    f = open(fname, 'w')
    f.write(str(fcontent))
    f.close()


#Get the sentinel bands needed for running the service
def get_sentinel_datacubes_for_cropsar(eoconn, biopar_type = 'FAPAR',biopar_fixed = False):
    # load the datacubes needed for cropsar calculation

    gamma0 = eoconn.load_collection('SENTINEL1_GAMMA0_SENTINELHUB', bands=['VH','VV'], properties  ={"polarization": lambda od: eq(od, "DV")}).sar_backscatter()


    # S2mask = create_mask(self._eoconn, scl_layer_band='SENTINEL2_L2A_SENTINELHUB:SCL')
    S2_bands = eoconn.load_collection('SENTINEL2_L2A_SENTINELHUB',
                                      bands=["B03", "B04", "B08", "sunAzimuthAngles", "sunZenithAngles",
                                             "viewAzimuthMean", "viewZenithMean", "SCL"])
    S2_bands_mask = S2_bands.process("mask_scl_dilation", data=S2_bands,
                                     scl_band_name="SCL")

    if biopar_fixed:
        context = {"biopar": "{}".format(biopar_type)}
    else:
        context = {"biopar": {"from_parameter": "biopar_type"}}
    S2_bands_mask = S2_bands_mask.resample_cube_spatial(gamma0)
    udf = load_udf(Path(__file__).parent / 'biopar_udf.py')
    fapar_masked = S2_bands_mask.reduce_dimension(dimension="bands", reducer=PGNode(
        process_id="run_udf",
        data={"from_parameter": "data"},
        udf=udf,
        runtime="Python",
        context= context
    ))
    fapar_masked = fapar_masked.add_dimension('bands', label='band_0', type='bands')

    return gamma0, fapar_masked


def preprocessing_datacube_for_udf(merged_cube, time_range, geo, bbox):
    """
    Preprocessing of the script includes in this case applying a data shift and inwards buffering
    :param merged_cube: the merged datacube containing all data layers needed for running the service
    :param time_range: the period for which the data should be requested
    :param geo: the geometry for which the data should be requested
    :param bbox: indicate whether a bbox should be used for requesting the data, which allows downloading the bbox extent
    :return: datacube containing all input data needed for running the service
    """
    if not bbox:
        udf_input_timeseries: DataCube = merged_cube.filter_temporal(start_date=time_range[0],end_date=time_range[1]).aggregate_spatial(geo, reducer='mean')
    else:
        udf_input_timeseries: DataCube = merged_cube.filter_temporal(start_date=time_range[0],end_date=time_range[1]).filter_bbox(geo)
    return udf_input_timeseries


def get_input_datacube_udf_cropsar(eoconn, time_range, geo, biopar_type = 'FAPAR',bbox = False, biopar_fixed = False):
    gamma0, fapar_masked = get_sentinel_datacubes_for_cropsar(eoconn, biopar_type = biopar_type, biopar_fixed = biopar_fixed)
    merged_cube = gamma0.merge_cubes(fapar_masked)
    input_datacube = preprocessing_datacube_for_udf(merged_cube, time_range, geo, bbox)
    return input_datacube