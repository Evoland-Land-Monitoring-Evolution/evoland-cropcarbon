"""
Pre-processing functions applied in OpenEO
"""


from openeo.rest.datacube import DataCube
from openeo.processes import if_
from openeo.internal.graph_building import PGNode
from pathlib import Path
from cropcarbon.openeo.masking import scl_mask_erode_dilate
from cropcarbon.openeo.cropsar_inputs import get_input_datacube_udf_cropsar



def load_udf(udf):
    with open(Path(__file__).parent / udf, 'r+', encoding= "utf8") as f:
        return f.read()


def add_meteo(connection, METEO_collection, other_bands, bbox,
              start, end, preprocess=False,
              target_crs=None):
    # AGERA5
    meteo = connection.load_collection(
        METEO_collection,
        spatial_extent=bbox,
        bands=['temperature-max', 
               'temperature-min',
               'temperature-mean',
               'solar-radiation-flux',
               'dewpoint-temperature',
               'vapour-pressure'],
        temporal_extent=[start, end]
    )

    if (target_crs is not None):
        meteo = meteo.resample_spatial(projection=target_crs, resolution=10.0)

    if preprocess:
        # Composite to dekads
        meteo = meteo.aggregate_temporal_period(period="dekad",
                                                reducer="mean")

        # Linearly interpolate missing values.
        # Shouldn't exist in this dataset but is good practice to do so
        meteo = meteo.apply_dimension(dimension="t",
                                    process="array_interpolate_linear")

    # Rename band to match Radix model requirements
    meteo = meteo.rename_labels('bands', ['temperature_max', 
                                          'temperature_min',
                                          'temperature_mean',
                                          'solar_radiation_flux',
                                          'temperature_dewpoint',
                                          'vapour_pressure'])

    # --------------------------------------------------------------------
    # Merge cubes
    # or return just meteo
    # --------------------------------------------------------------------
    if other_bands is None:
        return meteo

    merged_inputs = other_bands.merge_cubes(meteo)

    return merged_inputs


def add_SSM(connection, SSM_collection, other_bands, bbox,
              start, end, preprocess=False,
              target_crs=None):
    
    
    # SSM
    SSM = connection.load_collection(
        SSM_collection,
        spatial_extent=bbox,
        temporal_extent=[start, end]
    )

    if (target_crs is not None):
        SSM = SSM.resample_spatial(projection=target_crs, resolution=10.0)

    if preprocess:
        # Composite to dekads
        SSM = SSM.aggregate_temporal_period(period="dekad",
                                                reducer="mean")

        # Linearly interpolate missing values.
        # Shouldn't exist in this dataset but is good practice to do so
        SSM = SSM.apply_dimension(dimension="t",
                                    process="array_interpolate_linear")

    # --------------------------------------------------------------------
    # Merge cubes
    # or return just SSM
    # --------------------------------------------------------------------
    if other_bands is None:
        return SSM

    merged_inputs = other_bands.merge_cubes(SSM)

    return merged_inputs


def add_CROPSAR(connection, other_bands, geo,
              start, end):
    
    
    # get first cube with input data for cropsar
    time_range = (start, end)
    TS_cube = get_input_datacube_udf_cropsar(connection, 
                                             time_range, 
                                             geo,
                                             biopar_fixed = True)

    # load UDF
    cropsar_code = load_udf("cropsar_udf.py")
    CROPSAR = TS_cube.process("run_udf", data=TS_cube, 
                                        udf=cropsar_code, 
                                        runtime="Python",
                                        context = {"date": time_range})
    # --------------------------------------------------------------------
    # return just CROPSAR
    # --------------------------------------------------------------------

    return CROPSAR

def add_S2(connection, S2_collection, bbox,
              start, end, masking, 
              target_crs=None,
              preprocess=True,
              other_bands = None, 
              **processing_options):

    S2_bands = ["B03", "B04", "B08", 
                "sunAzimuthAngles", "sunZenithAngles",
                "viewAzimuthMean", "viewZenithMean"]
    if masking not in ['mask_scl_dilation', 'satio', None]:
        raise ValueError(f'Unknown masking option `{masking}`')
    if masking in ['mask_scl_dilation']:
        # Need SCL band to mask
        S2_bands.append("SCL")
    bands = connection.load_collection(
        S2_collection,
        bands=S2_bands,
        spatial_extent=bbox,
        temporal_extent=[start, end],
        max_cloud_cover=95
    )

    # NOTE: currently the tunings are disabled.
    #
    # temporal_partition_options = {
    #     "indexreduction": 2,
    #     "temporalresolution": "ByDay",
    #     "tilesize": 1024
    # }
    # bands.result_node().update_arguments(
    #     featureflags=temporal_partition_options)


    if (target_crs is not None):
        bands = bands.resample_spatial(projection=target_crs, resolution=10.0)

    # NOTE: For now we mask again snow/ice because clouds
    # are sometimes marked as SCL value 11!
    if masking == 'mask_scl_dilation':
        # TODO: double check cloud masking parameters
        # https://github.com/Open-EO/openeo-geotrellis-extensions/blob/develop/geotrellis-common/src/main/scala/org/openeo/geotrelliscommon/CloudFilterStrategy.scala#L54  # NOQA
        bands = bands.process(
            "mask_scl_dilation",
            data=bands,
            scl_band_name="SCL",
            kernel1_size=17, kernel2_size=77,
            mask1_values=[2, 4, 5, 6, 7],
            mask2_values=[3, 8, 9, 10, 11],
            erosion_kernel_size=3).filter_bands(
            bands.metadata.band_names[:-1])
    elif masking == 'satio':
        # Apply satio-based mask
        mask = scl_mask_erode_dilate(
            connection,
            bbox,
            scl_layer_band=S2_collection + ':SCL',
            target_crs=target_crs).resample_cube_spatial(bands)
        bands = bands.mask(mask)
    
    # Avoid high reflectance values of 21 000 ending 
    # up in the temporal aggregation. This is currenlty 
    # only an issue of Terrascope data, should be checked 
    # whether it also occurs for other provider!!
    # we want to mask all corrupt reflectance of 21 000
    bands = bands.apply(lambda x: (if_(x !=21000, x)))

    
    if processing_options.get('biopar', None) is not None:
        # Calculate the FAPAR from the original bands
        context = {"biopar": processing_options.get('biopar', None)}
        
        udf = load_udf(Path(__file__).parent / 'biopar_udf.py')
        fapar_band = bands.reduce_dimension(dimension="bands", reducer=PGNode(
            process_id="run_udf",
            data={"from_parameter": "data"},
            udf=udf,
            runtime="Python",
            context= context
        ))
        
        fapar_band = fapar_band.add_dimension('bands', 
                    label=processing_options.get('biopar', None),
                    type='bands')

        if processing_options.get('biopar_only'):
            bands = fapar_band
        else:
            bands = bands.merge_cubes(fapar_band)

    if preprocess:
        # Composite to dekads
        bands = bands.aggregate_temporal_period(period="dekad",
                                                reducer="median")

        # TODO: if we would disable it here, nodata values
        # will be 65535 and we need to cope with that later
        # Linearly interpolate missing values
        bands = bands.apply_dimension(dimension="t",
                                      process="array_interpolate_linear")
        
    # --------------------------------------------------------------------
    # Merge cubes
    # or return just S2 
    # --------------------------------------------------------------------
    if other_bands is None:
        return bands
    
    merged_inputs = other_bands.merge_cubes(bands)

    return merged_inputs



def gpp_preprocessed_inputs(
        connection, bbox, start: str, end: str,
        S2_collection=None,
        METEO_collection=None,
        SSM_collection=None,
        CROPSAR_collection=None,
        preprocess=True,
        geo = None,
        masking='mask_scl_dilation',
        **processing_options) -> DataCube:
    """Main method to get preprocessed inputs from OpenEO for
    downstream gpp calibration.

    Args:
        connection: OpenEO connection instance
        bbox (_type_): _description_
        start (str): Start date for requested input data (yyyy-mm-dd)
        end (str): Start date for requested input data (yyyy-mm-dd)
        S2_collection (str, optional): Collection name for S2 data.
                        Defaults to
                        None.
        METEO_collection (str, optional): Collection name for
                        meteo data. Defaults to None.
        SSM_collection (str, optional): _description_.
                        Defaults to None.
        CROPSAR_collection (str, optional): _description_.
                            Defaults to None.
        preprocess (bool, optional): Apply compositing and interpolation.
                        Defaults to True.
        geo (_type_): geometry if in the input retrieval already an 
                      aggregation should be done
        masking (str, optional): Masking method to be applied.
                                One of ['satio', 'mask_scl_dilation', None]
                                Defaults to 'mask_scl_dilation'.

    Returns:
        DataCube: OpenEO DataCube wich the requested inputs
    """
    target_crs = processing_options.get("target_crs", None)
    # should be first None
    bands = None
    # --------------------------------------------------------------------
    # Optical data
    # --------------------------------------------------------------------
    if S2_collection is not None and S2_collection:
        bands = add_S2(connection, S2_collection,
                          bbox, start, end, masking,
                          preprocess= preprocess,
                          other_bands=bands,
                          **processing_options)

   
    # --------------------------------------------------------------------
    # AGERA5 Meteo data
    # --------------------------------------------------------------------
    if METEO_collection is not None and METEO_collection:
        bands = add_meteo(connection, METEO_collection,
                          bands, bbox, start, end,
                          target_crs=target_crs)

    # --------------------------------------------------------------------
    # SSM data
    # --------------------------------------------------------------------
    if SSM_collection is not None and SSM_collection:
        bands = add_SSM(connection, SSM_collection,
                        bands, bbox, start, end,
                        target_crs=target_crs)

    # --------------------------------------------------------------------
    # CropSAR data
    # --------------------------------------------------------------------
    if CROPSAR_collection is not None and CROPSAR_collection:
        bands = add_CROPSAR(connection,
                        bands, geo, start, end)
    else:
        bands = bands.filter_temporal(start, end)
    
    return bands
