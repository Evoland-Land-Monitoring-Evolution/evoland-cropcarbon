""""
UDF used to retrieve just the FAPAR band from the datacube
and to allow merging with the GDMP max cube which is in 256x256 resolution
after applying the UDF
"""

from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    from openeo.udf import inspect
    ds = cube.get_array()
    inspect(data=[ds.dims], message="UDF logging dims for FAPAR selection")
    inspect(data=[ds.shape], message="UDF logging shape for FAPAR selection")
    orig_dims = ds.dims
    # drop bands from orig_dims
    orig_dims = list(orig_dims)
    orig_dims.remove('bands')
    xr_FAPAR = ds.sel(bands='FAPAR').copy()
    xr_FAPAR.values = ds.sel(bands='FAPAR').values
    xr_FAPAR = xr_FAPAR.drop('bands')
    # revert to original dimensions
    ds_out = xr_FAPAR.transpose(*orig_dims)
    inspect(data=[ds_out.shape], message="UDF logging shape for FAPAR selection output")
    inspect(data=[ds_out.dims], message="UDF logging dims for FAPAR selection output")
    
    return XarrayDataCube(ds_out)
