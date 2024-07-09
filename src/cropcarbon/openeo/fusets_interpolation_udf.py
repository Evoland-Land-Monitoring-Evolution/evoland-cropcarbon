"""
UDF used to apply satio interpolation on the timeseries data
"""
import sys
from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube
from openeo.udf import inspect


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply the satio interpolation on the timeseries data
    :param cube: XarrayDataCube
    :param context: Dict
    :return: XarrayDataCube
    """
    # first load the path from where the repo functions can be loaded
    sys.path.insert(0, 'tmp/venv_cropcarbon')
    sys.path.insert(0, 'tmp/venv_dep')
    from fusets import WhittakerTransformer

    # load relevant context
    ds = cube.get_array()
    # load the original dims
    orig_dims = ds.dims
    # reorder dimensions
    ds = ds.transpose('bands', 't', 'y', 'x')
    inspect(data=[ds.shape], message="UDF logging shape of my cube")
    # apply interpolation
    ds_interpol = WhittakerTransformer().fit_transform(ds,
                                                       smoothing_lambda=100)
    # convert values below zero to zero
    ds_interpol = ds_interpol.where(ds_interpol > 0, 0)
    # And make sure we revert back to original dimension order
    ds_interpol = ds_interpol.transpose(*orig_dims)
    return XarrayDataCube(ds_interpol)