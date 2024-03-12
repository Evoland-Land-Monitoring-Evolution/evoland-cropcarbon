""""
Script that will retrieve for the defined satellite datasets
timeseries for the selected grid of the considered ICOS sites
on grassland and cropland.
Dataset needs: 
- fAPAR aggregated and interpolated (cfr. cropclass preprocessing)
- SSM from S1
- AGERA5
"""


# import needed packages
import glob
from pathlib import Path
from cropcarbon.gpp.constants import basefold_extr
from loguru import logger
import json


def main(basedir, platform):
    # Get datasets
    datasets = glob.glob(str(basedir / '*'))
    datasets = [Path(d).name for d in datasets
                if (Path(d).is_dir() and Path(d).name != 'extraction_status')]

    logger.info(f'Found {len(datasets)} datasets to process.')

    # Get generic extraction settings
    settingsfile = str(basedir / f'Generic_settings_{platform}.json')
    settings = json.load(open(settingsfile, 'r'))

    # Get source of extractions
    source = settings.get('source')

    if source == 'openeo':
        # first create a feature collection of
        # the datasets that should be used for
        # extraction and info on which datasets
        # still need to be processed
        from cropcarbon.openeo.extractions import json_batch_jobs
        for dataset in datasets:
            json_batch_jobs(basedir, dataset, settingsfile)
        # get data provider from settings
        provider = settings.get('provider')

        from cropcarbon.openeo.extractions import main as extractions_f
        logger.info('Starting extractions using openeo...')
        extractions_f(basedir, provider)
    elif source == 'cds':
        from cropcarbon.cds.extractions import main as extractions_f
        logger.info('Starting extractions using cds...')
        extractions_f(basedir, datasets, settings)

    elif source == "edo":
        from cropcarbon.edo.extractions import main as extractions_f
        logger.info('Starting extractions using cds...')
        extractions_f(basedir, datasets, settings)

    elif source == 'GEE':
        raise ValueError(f'Source {source} not supported yet!')
    elif source == 'local_tiles':
        raise ValueError(f'Source {source} not supported yet!')
    else:
        raise ValueError(f'Source {source} not supported!')

    logger.success('All done!')


if __name__ == "__main__":
    platform = "edo" # OPTIONS: openeo, cds, edo
    basedir = Path(basefold_extr)

    main(basedir, platform)