"""
Pre-processing functions applied in OpenEO
"""

import openeo
from openeo.rest.datacube import DataCube
from openeo.processes import if_, or_
from openeo.internal.graph_building import PGNode
from pathlib import Path
from cropcarbon.openeo.masking import scl_mask_erode_dilate
from cropcarbon.openeo.cropsar_inputs import get_input_datacube_udf_cropsar


def load_udf(udf):
    with open(Path(__file__).parent / udf, 'r+', encoding= "utf8") as f:
        return f.read()


def add_LC(connection, LANDCOVER_collection, bbox,
           target_crs=None,
           other_dataset=None,
           mask_LC=False,
           bands_focus=None):
    
    # LAND COVER
    LC = connection.load_collection(
        LANDCOVER_collection,
        spatial_extent=bbox,
        temporal_extent=["2020-12-30", "2022-01-01"]
    )

    if (target_crs is not None):
        LC = LC.resample_spatial(projection=target_crs, resolution=10.0)
        
    if bands_focus is not None:
        LC_band = LC.filter_bands(bands_focus)
    if mask_LC and not other_dataset:
        # Apply mask on crop and grassland classes
        # only keep values in the band equal
        # to 30 (Grassland) or 40 (Cropland)
        LC_mask = ~ ((LC_band == 30) | (LC_band == 40))
        # mask the entire cube
    if mask_LC and other_dataset:
        # Apply mask on crop and grassland classes
        # only keep values in the band equal
        # to 30 (Grassland) or 40 (Cropland)
        
        # first resample to the same spatial resolution
        LC_band = LC_band.resample_cube_spatial(other_dataset,
                                                method='near')
        # collection has timestamps which we need to get rid of
        LC_band = LC_band.max_time()
        LC_mask = ~ LC_band.apply(lambda x: (or_(x == 30, x == 40)))
                
    # rename the label of the band
    LC_band = LC_band.rename_labels('bands', ['LC'])
      
    # --------------------------------------------------------------------
    # Merge cubes
    # or return just LC
    # --------------------------------------------------------------------
    if other_dataset is None:
        if mask_LC:
            LC_band = LC_band.mask(LC_mask)
        return LC_band
    if mask_LC:
        # need to multiply due to bug in openeo
        # with corrupt datatypes --> should be solved
        other_dataset = other_dataset.apply(lambda x: x*1)
        other_dataset = other_dataset.mask(LC_mask)
    merged_inputs = other_dataset.merge_cubes(LC_band)

    return merged_inputs


def add_meteo(connection, METEO_collection, bbox,
              start, end, preprocess=False,
              target_crs=None, other_dataset=None,
              bands_focus=[
                    'temperature-max', 'temperature-min',
                    'temperature-mean', 'solar-radiation-flux',
                    'dewpoint-temperature', 'vapour-pressure']):
    # AGERA5
    meteo = connection.load_collection(
        METEO_collection,
        spatial_extent=bbox,
        bands=bands_focus,
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
    renamed_bands = [item.replace('-', '_') for item in bands_focus]
    meteo = meteo.rename_labels('bands', renamed_bands)
    
    # --------------------------------------------------------------------
    # Merge cubes
    # or return just meteo
    # --------------------------------------------------------------------
    if other_dataset is None:
        return meteo

    merged_inputs = other_dataset.merge_cubes(meteo)

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
           other_dataset=None,
           **processing_options):

    S2_bands = ["B03", "B04", "B08", 
                "sunAzimuthAngles", "sunZenithAngles",
                "viewAzimuthMean", "viewZenithMean"]
    if masking not in ['mask_scl_dilation', 'satio', None]:
        raise ValueError(f'Unknown masking option `{masking}`')
    if masking in ['mask_scl_dilation']:
        # Need SCL band to mask
        S2_bands.append("SCL")
    if S2_collection == "SENTINEL2_L2A" or S2_collection == "SENTINEL2_L2A_SENTINELHUB": # NOQA
        bands = connection.load_collection(
            S2_collection,
            bands=S2_bands,
            spatial_extent=bbox,
            temporal_extent=[start, end],
            max_cloud_cover=95
        )
        scl_band_name = "SCL"
    elif S2_collection == "TERRASCOPE_S2_FAPAR_V2":
        bands = connection.load_collection(
            S2_collection,
            spatial_extent=bbox,
            temporal_extent=[start, end]
        )
        scl_band_name = "SCENECLASSIFICATION_20M"

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
        bands = bands.process("mask_scl_dilation",
                              data=bands,
                              scl_band_name=scl_band_name).filter_bands(
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
    bands = bands.apply(lambda x: (if_(x != 21000, x)))
    
    if processing_options.get('biopar', None) is not None and S2_collection != "TERRASCOPE_S2_FAPAR_V2":
        # Calculate the FAPAR from the original bands
        context = {"biopar": processing_options.get('biopar', None)}
        
        udf = openeo.UDF.from_file(Path(__file__).parent / 'biopar_udf.py',
                                   context=context)
        fapar_band = bands.reduce_dimension(dimension="bands",
                                            reducer=udf)
        
        fapar_band = fapar_band.add_dimension('bands',
                                              label=processing_options.get('biopar', None), # NOQA
                                              type='bands')

        if processing_options.get('biopar_only'):
            bands = fapar_band
        else:
            bands = bands.merge_cubes(fapar_band)
    elif S2_collection == "TERRASCOPE_S2_FAPAR_V2":
        # rename band to proper name
        bands = bands.rename_labels('bands', ['FAPAR'])
        # apply scale factor to bands
        bands = bands.apply(lambda x: x*0.005)
    if preprocess:
        # Composite to dekads
        bands = bands.aggregate_temporal_period(period="dekad",
                                                reducer="median")

        # TODO: if we would disable it here, nodata values
        # will be 65535 and we need to cope with that later
        # Linearly interpolate missing values
        if processing_options.get("interpolation", None) is None:
            bands = bands
        elif processing_options.get("interpolation", None) == "satio":
            # Apply satio-based interpolation
            bands = bands.apply_dimension(
                openeo.UDF.from_file(Path(__file__).parent / "satio_interpolation_udf.py"), # NOQA
                dimension="t")
        elif processing_options.get("interpolation", None) == "fuseTS":
            bands = bands.apply_dimension(
                openeo.UDF.from_file(Path(__file__).parent / "fusets_interpolation_udf.py"), # NOQA
                dimension="t")
            
        elif processing_options.get("interpolation", None) == "linear":
            bands = bands.apply_dimension(dimension="t",
                                          process="array_interpolate_linear")
        else:
            raise ValueError(f'INTERPOLATION METHOD {processing_options.get("interpolation", None)} NOT YET SUPPORTED') # NOQA
            
    # --------------------------------------------------------------------
    # Merge cubes
    # or return just S2 
    # --------------------------------------------------------------------
    if other_dataset is None:
        return bands
    
    merged_inputs = other_dataset.merge_cubes(bands)

    return merged_inputs


def gpp_preprocessed_inputs(
        connection, bbox, start: str, end: str,
        S2_collection=None,
        METEO_collection=None,
        SSM_collection=None,
        CROPSAR_collection=None,
        LANDCOVER_collection=None,
        preprocess=True,
        geo=None,
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
        LANDCOVER_collection (str, optional): _description_.
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
                       preprocess=preprocess,
                       other_bands=bands,
                       **processing_options)

    # --------------------------------------------------------------------
    # AGERA5 Meteo data
    # --------------------------------------------------------------------
    if METEO_collection is not None and METEO_collection:
        if processing_options.get('METEO_BANDS', None) is not None:
            bands = add_meteo(connection, METEO_collection,
                              bbox, start, end,
                              target_crs=target_crs,
                              other_dataset=bands,
                              bands_focus=processing_options.get('METEO_BANDS',
                                                                 None))
        else:
            bands = add_meteo(connection, METEO_collection,
                              bbox, start, end,
                              target_crs=target_crs,
                              other_dataset=bands)

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
        bands = add_CROPSAR(connection, bands, geo, start, end)
        
    # --------------------------------------------------------------------
    # LAND COVER data
    # --------------------------------------------------------------------
    if LANDCOVER_collection is not None and LANDCOVER_collection:
        bands = add_LC(connection, LANDCOVER_collection, bbox,
                       target_crs=target_crs,
                       other_dataset=bands,
                       mask_LC=processing_options.get('MASK_LC', None),
                       bands_focus=processing_options.get('LC_BANDS', None))
    # apply a final filtering on bbox and time period
    bands = bands.filter_bbox(bbox)#.filter_temporal(start, end)
    # add now the bands dimension again
    return bands
