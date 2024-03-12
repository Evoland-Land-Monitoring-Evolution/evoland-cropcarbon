"""
Code that will prepare the extracted EO data
to be used for calibrating of the GPP model.

The following pre-processing steps should be conducted:
- Apply some interpolation on the FAPAR data
- Bring the in-situ and EO data on the same temporal scale (dekad, month)
- Retain only the dekads for which both in-situ and FAPAR/METEO data are available
"""


# import needed packages
from pathlib import Path
from cropcarbon.gpp.constants import basefold_cal
from cropcarbon.utils.storage import _make_serial
from loguru import logger
import json
import os




def main(basedir):
    # Get generic settings for preparing data
    settingsfile = str(basedir / 'Generic_settings.json')
    settings = json.load(open(settingsfile, 'r'))

    from cropcarbon.gpp.calibration.inputs import main as inputs_f

    logger.info('Starting GPP inputs preparation...')
    inputs_f(basedir, settings)

    # write out the used settings for preparing the data
    target_dir = basedir.joinpath('data',
                                  settings.get('VERSION'))
    meta_name = f'metadata_{settings.get("VERSION")}.json'
    if not os.path.exists(target_dir.joinpath(meta_name))\
        or settings.get("overwrite"):
        with open(target_dir / meta_name, 'w') as f:
                json.dump(_make_serial(settings), f, ensure_ascii=False)

    logger.success('FINISHED SUCESSFULLY')

if __name__ == "__main__":

    main(Path(basefold_cal))