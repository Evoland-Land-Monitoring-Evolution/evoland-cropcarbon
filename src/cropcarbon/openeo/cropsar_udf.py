import logging
from openeo.udf.udf_data import UdfData
from openeo.udf.structured_data import StructuredData
from openeo.rest.conversions import timeseries_json_to_pandas
import pandas as pd
#import sys
#sys.path.append(r'/data/users/Public/bontek/Nextland/Anomaly_detection/cropsar-1.4.7-py3-none-any.whl') #TODO TO remove
from cropsar.preprocessing.retrieve_timeseries_openeo import run_cropsar_dataframes


logger = logging.getLogger("nextland_services.cropsar")


# calculate the cropsar curve for each field and the regional average of all the input fields
######## FUNCTIONS ################
def get_cropsar_TS(ts_df, unique_ids_fields, metrics_order, time_range, Spark=True):
    index_fAPAR = metrics_order.index('FAPAR')
    column_indices = ts_df.columns.get_level_values(1)
    indices = column_indices.isin([index_fAPAR])

    df_S2 = ts_df.loc[:, indices].sort_index().T
    if(df_S2.empty):
        raise ValueError("Received an empty Sentinel-2 input dataframe while trying to compute cropSAR!")

    df_VHVV = ts_df.loc[:, column_indices.isin([0, 1])].sort_index().T

    cropsar_df, cropsar_df_q10, cropsar_df_q90 = run_cropsar_dataframes(
        df_S2, df_VHVV, None, scale=1, offset=0, date_range=time_range,
    )

    if (len(cropsar_df.index) == 0):
        logger.warning("CropSAR returned an empty dataframe. For input, Sentinel-2 input: ")
        logger.warning(str(df_S2.to_json(indent=2)))
        logger.warning("Sentinel-1 VH-VV: ")
        logger.warning(str(df_VHVV.to_json(indent=2)))

    cropsar_df = cropsar_df.rename(
        columns=dict(zip(list(cropsar_df.columns.values), [str(item) + '_CROPSAR' for item in unique_ids_fields])))
    cropsar_df = cropsar_df.round(decimals=3)
    cropsar_df.index = pd.to_datetime(cropsar_df.index).date
    cropsar_df = cropsar_df.loc[pd.to_datetime(time_range[0], format = '%Y-%m-%d').date() :pd.to_datetime(time_range[1], format = '%Y-%m-%d').date()]
    return cropsar_df


def udf_cropsar(udf_data: UdfData):
    ## constants
    user_context = udf_data.user_context
    time_range = user_context.get('date')
    columns_order = ['VH', 'VV', 'FAPAR']

    ## load the TS
    ts_dict = udf_data.get_structured_data_list()[0].data
    if not ts_dict:  # workaround of ts_dict is empty
        return
    TS_df = timeseries_json_to_pandas(ts_dict)
    TS_df.index = pd.to_datetime(TS_df.index).date


    if not isinstance(TS_df.columns, pd.MultiIndex):
        TS_df.columns = pd.MultiIndex.from_product([[0], TS_df.columns])

    amount_fields = next(iter(ts_dict.values()))
    unique_ids_fields = ['Field_{}'.format(str(p)) for p in range(len(amount_fields))]


    logger.info("CropSAR input dataframe:\n" + str(TS_df.describe()))
    logger.info("Input time range: " + str(time_range))
    #logger.warning(str(TS_df.to_json(indent=2,date_format='iso',double_precision=5)))

    ts_df_cropsar = get_cropsar_TS(TS_df, unique_ids_fields, columns_order, time_range)
    ts_df_cropsar = ts_df_cropsar.round(decimals=3)
    ts_df_cropsar.index = ts_df_cropsar.index.astype(str)


    udf_data.set_structured_data_list(
        [StructuredData(description='cropsar', data=ts_df_cropsar.to_dict(), type="dict")])

    return udf_data