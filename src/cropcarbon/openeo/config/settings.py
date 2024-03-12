from pathlib import Path




# ---------------------------------------------------
# Job options for OpenEO inference
DEFAULT_JOB_OPTIONS = {
    "driver-memory": "1500m",
    "driver-memoryOverhead": "512m",
    "driver-cores": "1",
    "executor-memory": "850m",
    "executor-memoryOverhead": "2900m",
    "executor-cores": "1",
    "executor-request-cores": "600m",
    "max-executors": "22",
    "executor-threads-jvm": "7",
    "logging-threshold": "info",
    "mount_tmp": False,
    "goofys": "false",
    "node_caching": True
}


# Terrascope backend specific options
TERRASCOPE_JOB_OPTIONS = {
    "task-cpus": 1,
    "executor-cores": 1,
    "max-executors": 32,
    "queue": "default",
    # terrascope reads from geotiff instead of jp2, so no threading issue there
    "executor-threads-jvm": 12,
    "executor-memory": "2g",
    "executor-memoryOverhead": "3g"
}


# Sentinelhub layers specific options
SENTINELHUB_JOB_OPTIONS = {
    "sentinel-hub": {
        "client-alias": "vito"
    }
}

# ---------------------------------------------------
# Job options for OpenEO training data extractions

OPENEO_EXTRACT_JOB_OPTIONS = {
    "driver-memory": "4G",
    "driver-memoryOverhead": "4G",
    "driver-cores": "2",
    "executor-memory": "3G",
    "executor-memoryOverhead": "2G",
    "executor-cores": "2",
    "max-executors": "50",
    "soft-errors": "true"
}

OPENEO_EXTRACT_CREO_JOB_OPTIONS = {
    "driver-memory": "4G",
    "driver-memoryOverhead": "2G",
    "driver-cores": "1",
    "executor-memory": "2000m",
    "executor-memoryOverhead": "3500m",
    "executor-cores": "4",
    "executor-request-cores": "400m",
    "max-executors": "200"
}

# ---------------------------------------------------
# Collection options for OpenEO inference

# Collection definitions on Terrascope
_TERRASCOPE_COLLECTIONS = {
    'S2_collection': "SENTINEL2_L2A",
    'METEO_collection': 'AGERA5',
    'SSM_collection': 'CGLS_SSM_V1_EUROPE'
}

# Collection definitions on CREO
_CREO_COLLECTIONS = {
    'S2_collection': "SENTINEL2_L2A",
    'METEO_collection': None,
    'SSM_collection': None
}

# Collection definitions on Sentinelhub
_SENTINELHUB_COLLECTIONS = {
    'S2_collection': "SENTINEL2_L2A",
    'METEO_collection': None,
    'SSM_collection': None
}


def _get_default_job_options(task: str = 'inference'):
    if task == 'inference':
        return DEFAULT_JOB_OPTIONS
    elif task == 'extractions':
        return OPENEO_EXTRACT_JOB_OPTIONS
    else:
        raise ValueError(f'Task `{task}` not known.')


def get_job_options(provider: str = None,
                    task: str = 'inference'):

    job_options = _get_default_job_options(task)

    if task == 'inference':
        if provider.lower() == 'terrascope':
            job_options.update(TERRASCOPE_JOB_OPTIONS)
        elif provider.lower() == 'sentinelhub' or provider.lower() == 'shub':
            job_options.update(SENTINELHUB_JOB_OPTIONS)
        elif provider.lower() == 'creodias':
            pass
        elif provider is None:
            pass
        else:
            raise ValueError(f'Provider `{provider}` not known.')

    elif task == 'extractions':
        if provider.lower() == 'creodias':
            job_options.update(OPENEO_EXTRACT_CREO_JOB_OPTIONS)

    return job_options




def get_collection_options(provider: str):

    if provider.lower() == 'terrascope':
        return _TERRASCOPE_COLLECTIONS
    elif provider.lower() == 'sentinelhub' or provider.lower() == 'shub':
        return _SENTINELHUB_COLLECTIONS
    elif 'creo' in provider.lower():
        return _CREO_COLLECTIONS
    else:
        raise ValueError(f'Provider `{provider}` not known.')
