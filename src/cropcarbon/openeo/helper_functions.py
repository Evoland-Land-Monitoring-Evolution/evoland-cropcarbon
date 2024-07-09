"""
Collection of some relevant helper functions needed
to operationalize and test a workflow in OpenEO
"""

# import needed packages
import xarray as xr
import os
import openeo
from openeo.udf import XarrayDataCube
from cropcarbon.openeo.preprocessing import gpp_preprocessed_inputs
from openeo.udf import execute_local_udf
from pathlib import Path
from fusets import WhittakerTransformer


def test_debug_interpolation(dir_data, interpolation_method='satio'):
    ds = xr.load_dataset(dir_data,
                         engine='h5netcdf').drop('crs').to_array(dim='bands')
    if interpolation_method == 'satio':
        udf_interpolation = Path(os.path.join(Path(__file__).parent,
                                              'satio_interpolation_udf.py')).read_text() # NOQA
        from cropcarbon.openeo.satio_interpolation_udf import apply_datacube
    elif interpolation_method == 'fuseTS':
        res = WhittakerTransformer().fit_transform(ds)
         
    else:
        raise ValueError(f'Interpolation method {interpolation_method}'
                         ' not supported yet!')
    # first test if no errors are present in the UDF itself
    res = execute_local_udf(udf_interpolation, XarrayDataCube(ds))
    # second step-in to the UDF to check the actual process
    res = apply_datacube(XarrayDataCube(ds), {})
    return res


def test_calc_GDMP_max(dir_data, config):
    # load the UDF for the GPP calculation
    # udf_gdmp_max = Path(os.path.join(Path(__file__).parent,
    #                             'GDMP_max_estimation_udf.py')).read_text()
    from cropcarbon.openeo.f_CO2_estimation_udf import apply_datacube as a1
    from cropcarbon.openeo.GDMP_max_estimation_udf import apply_datacube as a2
    from cropcarbon.openeo.get_FAPAR_band_udf import apply_datacube as a3
    # load the dataset
    ds = xr.load_dataset(dir_data,
                         engine='h5netcdf').drop('crs').to_array(dim='bands')
    # first test if no errors are present in the UDF itself
    # Warning cannot pass context to UDF!!!
    # res = execute_local_udf(udf_gdmp_max, XarrayDataCube(ds))
    # second step-in to the UDF to check the actual process
    res = a1(XarrayDataCube(ds), config)
    res = a2(XarrayDataCube(res.array), config)
    res = a3(XarrayDataCube(res.array), config)
    return res


def test_calc_GPP(dir_data, config):
    # load the UDF for the GPP calculation
    udf_gpp = Path(os.path.join(Path(__file__).parent,
                                'GPP_estimation_udf.py')).read_text()
    from cropcarbon.openeo.GPP_estimation_udf import apply_datacube as a1
    # load the dataset
    ds = xr.load_dataset(dir_data,
                         engine='h5netcdf').drop('crs').to_array(dim='bands')
    # first test if no errors are present in the UDF itself
    # Warning cannot pass context to UDF!!!
    # res = execute_local_udf(udf_gpp, XarrayDataCube(ds))
    # second step-in to the UDF to check the actual process
    res = a1(XarrayDataCube(ds), {})
    return res


def wrapper_GPP_est(connection, datasets, BBOX, start_date, end_date,
                    GPP_config,
                    projection=3035,
                    GDMP_max_only=False):
    # get the datacube for the GPP estimation
    cube_dataset = wrapper_get_datacube_gpp(connection,
                                            datasets,
                                            BBOX,
                                            start_date,
                                            end_date,
                                            projection=projection)
    # load ECO2 UDF
    ECO2_udf = Path(os.path.join(Path(__file__).parent,
                                 'f_CO2_estimation_udf.py')).read_text()
    # apply the UDF function
    cube_input_GDMP_MAX = cube_dataset.apply_neighborhood(
        lambda data: data.run_udf(udf=ECO2_udf, runtime='Python', context=GPP_config), # NOQA
        size=[
            {'dimension': 'x', 'value': 64, 'unit': 'px'},
            {'dimension': 'y', 'value': 64, 'unit': 'px'}
        ],
        overlap=[]
    )
    # rename the ECO2 band to the proper name
    target_band_names = cube_input_GDMP_MAX.metadata.band_names + ['ECO2']
    cube_input_GDMP_MAX = cube_input_GDMP_MAX.rename_labels(
        dimension='bands',
        target=target_band_names)
    # filter only on the ECO2 band
    ECO2_band = cube_input_GDMP_MAX.filter_bands(bands=['ECO2'])
    # merge it with the original cube
    cube_dataset_GDMP_MAX = cube_dataset.merge_cubes(ECO2_band)
        
    # load the GDMP max UDF
    GDMP_max_udf = openeo.UDF.from_file(os.path.join(Path(__file__).parent,
                                                     'GDMP_max_estimation_udf.py'), # NOQA
                                        context=GPP_config)
    # apply the UDF function
    GDMP_max_band = cube_dataset_GDMP_MAX.reduce_dimension(
        dimension="bands",
        reducer=GDMP_max_udf)
    GDMP_max_band = GDMP_max_band.add_dimension('bands',
                                                label='GDMP_max',
                                                type='bands')
    FAPAR_band = cube_dataset.filter_bands(bands=['FAPAR'])
    # merge FAPAR and GDMP max
    GDMP_max_inputs = FAPAR_band.merge_cubes(GDMP_max_band) # NOQA
  
    # Composite to dekads
    GDMP_max_inputs = GDMP_max_inputs.aggregate_temporal_period(period="dekad",
                                                                reducer="mean")
    if GDMP_max_only:
        return GDMP_max_inputs
    # load the GPP UDF
    GPP_udf = openeo.UDF.from_file(os.path.join(Path(__file__).parent,
                                                'GPP_estimation_udf.py'),
                                   context={})
    # apply the UDF function
    GPP_band = GDMP_max_inputs.reduce_dimension(
        dimension="bands",
        reducer=GPP_udf)
    GPP_band = GPP_band.add_dimension('bands',
                                      label='GPP',
                                      type='bands')
    return GPP_band


def wrapper_get_datacube_gpp(connection, datasets, BBOX, start_date, end_date,
                             projection=3035):
        
    if "S2_FAPAR" in datasets:
        S2_collection = datasets.get('S2_FAPAR').get('NAME')
        SCL_masking = datasets.get('S2_FAPAR').get('MASKING')
        # retrieve the interpolation method requested
        interpolation = datasets.get('S2_FAPAR').get('interpolation')
        biopar = 'FAPAR'
        biopar_only = True
    else:
        S2_collection = None
        biopar = None
        biopar_only = False
    if "AGERA5" in datasets:
        METEO_collection = datasets.get('AGERA5').get('NAME')
        METEO_BANDS = datasets.get('AGERA5').get('BANDS_FOCUS')
    else:
        METEO_collection = None
        METEO_BANDS = None
    if "LANDCOVER" in datasets:
        LANDCOVER_collection = datasets.get('LANDCOVER').get('NAME')
        # based on start date decide if the 2020 or 2021 version should be used
        if start_date < '2021-01-01':
            LANDCOVER_collection = LANDCOVER_collection.format('2020', 'V1')
        else:
            LANDCOVER_collection = LANDCOVER_collection.format('2021', 'V2')
        LC_BANDS = datasets.get('LANDCOVER').get('BANDS_FOCUS')
        MASK_LC = datasets.get('LANDCOVER').get('MASK_LC')
    else:
        LANDCOVER_collection = None
        LC_BANDS = None
        MASK_LC = None
    cube_dataset = gpp_preprocessed_inputs(connection,
                                           BBOX,
                                           start_date,
                                           end_date,
                                           S2_collection=S2_collection,
                                           METEO_collection=METEO_collection,
                                           LANDCOVER_collection=LANDCOVER_collection, # NOQA
                                           masking=SCL_masking,
                                           target_crs=projection,
                                           interpolation=interpolation,
                                           biopar=biopar,
                                           biopar_only=biopar_only,
                                           METEO_BANDS=METEO_BANDS,
                                           LC_BANDS=LC_BANDS,
                                           MASK_LC=MASK_LC)
    return cube_dataset