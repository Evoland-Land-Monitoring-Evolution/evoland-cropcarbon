"""
Functions to retrieve TS of flux
data from raw input
"""

import os
import pandas as pd
import glob
from loguru import logger as log
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict
import datetime

from satio.timeseries import Timeseries



def _get_TS_(df_sites, basefolder, outfolder,
             years):
    lst_subdirs = ([name for name in os.listdir(basefolder)
                    if os.path.isdir(os.path.join(basefolder, name))])

    for i, row in df_sites.iterrows():
        site_ID = row.siteid
        subdir_site = [item for item in lst_subdirs if site_ID in item]
        if len(subdir_site) > 1:
            log.warning(
                f'Multiple data matched found for site: {site_ID} --> check')
        if not subdir_site:
            continue

        fn_csv = r'ICOSETC_'+site_ID+'_FLUXNET_DD_L2.csv'
        fp_csv = os.path.join(basefolder, subdir_site[0], fn_csv)
        if os.path.isfile(fp_csv) is False:
            continue
        df_csv = pd.read_csv(fp_csv, index_col=0, parse_dates=True)
        cols_sample = ['GPP_NT_VUT_REF', 'GPP_DT_VUT_REF',
                       'GPP_NT_VUT_MEAN', 'GPP_DT_VUT_MEAN',
                       'GPP_NT_CUT_MEAN', 'GPP_DT_CUT_MEAN']
        cols_match = [item for item in cols_sample if item in list(
            df_csv.columns.values)]
        df_subset = df_csv[cols_match]

        df_subset = df_subset.mask(df_subset < -9000)
        for year in years:

            # check if data is not already present
            outfolder_year = os.path.join(outfolder, str(year))
            os.makedirs(outfolder_year, exist_ok=True)
            outname_csv = site_ID+f'_flx_gpp_{str(year)}_dd.csv'
            if not os.path.exists(os.path.join(outfolder_year, outname_csv)):

                # check if data is available
                # for that year according to the meta file

                av_year = row[f'av_{str(year)}']
                if av_year == 0:
                    continue

                df_subset_year = df_subset[f'{str(year)}-01-01':f'{str(year)}-12-31']  # NOQA
                df_subset_year.to_csv(os.path.join(
                    outfolder_year, outname_csv), index=True)


def aggregate_TS(df_TS: pd.DataFrame, start: str, end: str, period: str, reducer: str) -> pd.DataFrame:
    """"
    Function that allows to aggregate timeseries to a certain temporal scale (e.g., dekad, month,...) 

    :param df_TS: the dataframe with the timeseries on which the aggregation should be applied
    :param start: the start date for aggregation
    :param end: the end date for aggregation
    :param period: the temporal scale for aggregation (.e.g, dekad, month)
    :param reducer: the statistical way the data should be aggregated (.e.g., mean, median)
    :return: dataframe that contains the aggregated timeseries
    """
    from datetime import timedelta
    import calendar

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    intervals = []
    if "week" == period:
        offset = timedelta(weeks=1)
        start_dates = pd.date_range(
            start - offset, end, freq='W', inclusive='left')
        end_dates = pd.date_range(start, end + offset, freq='W', 
                                  inclusive='left')
        intervals = zip(start_dates, end_dates)
    elif "month" == period:
        offset = timedelta(weeks=4)
        start_dates = pd.date_range(
            start - offset, end, freq='MS', inclusive='left')
        end_dates = pd.date_range(
            start_dates[0] + timedelta(weeks=3), end + offset, freq='MS', 
            inclusive='left')
        intervals = zip(start_dates, end_dates)
    elif "year" == period:
        offset = timedelta(weeks=52)
        start_dates = pd.date_range(
            start - offset, end,
            freq='A-DEC', inclusive='left') + timedelta(days=1)
        end_dates = pd.date_range(
            start, end+offset, freq='A-DEC', inclusive='left') + timedelta(days=1)
        intervals = zip(start_dates, end_dates)
    elif "day" == period:
        offset = timedelta(days=1)
        start_dates = pd.date_range(
            start - offset, end, freq='D', inclusive='left')
        end_dates = pd.date_range(start, end + offset, freq='D',
                                  inclusive='left')
        intervals = zip(start_dates, end_dates)
    elif "dekad" == period:
        offset = timedelta(days=10)
        start_dates = pd.date_range(
            start - offset, end, freq='MS', inclusive='left')
        ten_days = pd.Timedelta(days=10)
        first_dekad_month = [(date, date+ten_days) for date in start_dates]
        second_dekad_month = [
            (date + ten_days, date + ten_days+ten_days) for date in start_dates]
        end_month = [date + pd.Timedelta(days=calendar.monthrange(
            date.year, date.month)[1]) for date in start_dates]

        third_dekad_month = list(
            zip([date + ten_days+ten_days for date in start_dates], end_month))
        intervals = (first_dekad_month + second_dekad_month + third_dekad_month)
        intervals.sort(key=lambda t: t[0])

    intervals = list([i for i in intervals if i[0] < end])
    lst_df_dekad_value = []
    # now do the actual aggregation
    for interval in intervals:
        middle_interval = (interval[0] + (interval[1] - interval[0])/2).date()
        interval_date_lst = [item.date() for item in list(interval)]
        df_aggreg_period = df_TS.loc[interval_date_lst[0]: interval_date_lst[1]]
        if 'mean' == reducer:
            df_aggreg_period = pd.DataFrame(df_aggreg_period.mean()).T
        elif 'median' == reducer:
            df_aggreg_period = pd.DataFrame(df_aggreg_period.median()).T
        elif 'mode' == reducer:
            df_aggreg_period = pd.DataFrame(df_aggreg_period.mode())
            # if still options remains, take the lowest category
            if df_aggreg_period.shape[0] > 1 and len(lst_df_dekad_value) > 0:
                last_class = lst_df_dekad_value[-1].values[0]
                if np.where(df_aggreg_period == last_class)[0].size == 0:
                    # take the first one as that one 
                    # is closer to the actual dekad date
                    df_aggreg_period = pd.DataFrame(df_aggreg_period.iloc[0]).T
                else:
                    col_name = df_TS.columns
                    df_aggreg_period = pd.DataFrame(data=last_class,
                                                    columns=col_name)
            else:
                # take the first one as that one 
                # is closer to the actual dekad date
                df_aggreg_period = pd.DataFrame(df_aggreg_period.iloc[0]).T

        df_aggreg_period.index = [interval_date_lst[0]]
        lst_df_dekad_value.append(df_aggreg_period)

    # concate the dataframes
    df_aggregated = pd.concat(lst_df_dekad_value)

    return df_aggregated


def cleaned_TS_agg(infolder, dict_periods,
                   column_id='siteid',
                   reducer='mean', overwrite=False):

    files = glob.glob(os.path.join(infolder, '*_cleaned_*.csv'))
    for file in files:
        df_TS = pd.read_csv(file, index_col=0,
                            parse_dates=True)

        sites = list(df_TS[column_id].unique())
        year_file = Path(file).stem.split('_')[-1]
        start = f'{str(year_file)}-01-01'
        end = f'{str(year_file)}-12-31'
        for period in dict_periods.keys():
            out_fold = os.path.join(dict_periods.get(period),
                                    'grouped')
            os.makedirs(out_fold, exist_ok=True)
            out_name = f'ICOS_GPP_{period}_{str(year_file)}_cleaned.csv'
            os.makedirs(out_fold, exist_ok=True)
            if os.path.exists(os.path.join(out_fold, out_name)) and not overwrite:
                continue
            lst_TS_aggr = []
            for site in sites:
                df_TS_site = df_TS.loc[df_TS[column_id] == site]

                df_agg_TS = aggregate_TS(df_TS_site, start, end,
                                         period, reducer=reducer)
                df_agg_TS.index.name = 'TIMESTAMP'
                df_agg_TS[column_id] = [site] * df_agg_TS.shape[0]
                lst_TS_aggr.append(df_agg_TS)

            df_TS_period_grouped = pd.concat(lst_TS_aggr)
            df_TS_period_grouped.to_csv(os.path.join(out_fold, out_name),
                                        index=True)


def wrapper_TS_agg(infolder, dict_periods,
                   reducer='mean', overwrite=False):
    # check files in folder

    TS_files_dd = glob.glob(os.path.join(infolder,
                                         '**', '*.csv'))
    TS_files_dd = [item for item in TS_files_dd if not 'grouped' in item]

    for file in TS_files_dd:
        df_TS = pd.read_csv(file, index_col=0,
                            parse_dates=True)
        year_file = Path(file).parent.stem
        start = f'{str(year_file)}-01-01'
        end = f'{str(year_file)}-12-31'

        for period in dict_periods.keys():
            out_fold = os.path.join(dict_periods.get(period),
                                    year_file)
            out_name = Path(file).stem.replace('_dd', f'_{period}.csv')
            if os.path.exists(os.path.join(out_fold, out_name)) and not overwrite:
                continue
            os.makedirs(out_fold, exist_ok=True)
            df_agg_TS = aggregate_TS(df_TS, start, end,
                                     period, reducer=reducer)
            df_agg_TS.index.name = 'TIMESTAMP'
            df_agg_TS.to_csv(os.path.join(out_fold,
                                          out_name),
                             index=True)


def clean_TS(df):
    # negative values should be removed
    df_subset = df.mask(df < 0)

    # check if constant values
    # do occur in the column

    #df_subset[df_subset.diff() == 0].shift(-1).eq(0).sum()
    df_subset[df_subset[df_subset.diff() == 0].shift(-1).eq(0)] = np.nan

    return df_subset


def concat_TS(in_folder, cleaning=True, overwrite=False):
    years = os.listdir(in_folder)
    years = [item for item in years if not 'grouped' in item]
    out_folder = os.path.join(in_folder,
                              'grouped')
    os.makedirs(out_folder, exist_ok=True)

    for year in years:
        outname_or = f'ICOS_GPP_dd_{str(year)}.csv'
        outname_cleaned = f'ICOS_GPP_dd_cleaned_{str(year)}.csv'

        if os.path.exists(os.path.join(out_folder, outname_cleaned)) and not overwrite:
            continue

        files_year = glob.glob(os.path.join(in_folder,
                                            str(year),
                                            '*.csv'))
        lst_df_cleaned = []
        lst_df_or = []
        for file in files_year:
            siteid = Path(file).stem.split('_')[0]
            df_TS = pd.read_csv(file,
                                index_col=0,
                                parse_dates=True)
            df_TS_cleaned = clean_TS(df_TS)

            df_TS['siteid'] = [siteid] * df_TS.shape[0]
            lst_df_or.append(df_TS)
            df_TS_cleaned['siteid'] = [siteid] * df_TS_cleaned.shape[0]
            lst_df_cleaned.append(df_TS_cleaned)

        # now concatenate everything
        df_all_or = pd.concat(lst_df_or)
        df_all_cleaned = pd.concat(lst_df_cleaned)

        df_all_or.to_csv(os.path.join(out_folder,
                                      outname_or),
                         index=True)

        df_all_cleaned.to_csv(os.path.join(out_folder,
                                           outname_cleaned),
                              index=True)



def to_satio_timeseries(ds: xr.DataArray, sensor: str):
    """
    Method to convert the input dataframe in a
    format that can be handeled with the satio
    """
    log.info('Transforming df into satio.Timeseries ...')

    satio_ts = {}
    
    # Transform the DataArray to satio Timeseries
    ts = Timeseries(
        data=ds.values,
        timestamps=list(ds.coords['t'].values),
        bands=list(ds.coords['bands'].values)
    )
    satio_ts[sensor] = ts
 
    return ts


def nc_to_pandas(data, bands, year_start, 
                year_end, dict_collection,
                collection_name, clip_year=False,
                rename_bands=False, translate_bands=None):

    collection_data = data[bands]

    for year in np.arange(int(year_start), 
                                int(year_end)+1,1):
        # per year a seperate frame should be created
        lst_df_collection_yr = []
        if not clip_year:
            # take 6 months around to allow later on interpolation
            if type(collection_data.t.values[0]) == datetime.date:
                collection_year = collection_data.sel(t=slice(datetime.date(int(year-1), 7, 1),
                                                datetime.date(int(year+1), 7, 1)))
            else:
                collection_year = collection_data.sel(t=slice(f'{str(year-1)}-07-01',
                                                f'{str(year+1)}-07-01'))
        else:
            if type(collection_data.t.values[0]) == datetime.date:
                collection_year = collection_data.sel(t=slice(datetime.date(int(year), 1, 1),
                                                datetime.date(int(year), 12, 31)))
            else:
                collection_year = collection_data.sel(t=slice(f'{str(year)}-01-01',
                                                f'{str(year)}-12-31'))
        for band in bands:
            data_band = collection_year[band]
            if data_band.t.shape != data_band.data.shape:
                df_band_year = pd.DataFrame(np.reshape(data_band.data, 
                                                       data_band.t.shape), 
                                            columns=[band], 
                                            index=data_band.t)
            else:
                df_band_year = pd.DataFrame(data_band.data, 
                                            columns=[band], 
                                            index=data_band.t)
            if rename_bands:
                df_band_year = df_band_year.rename(columns={band: translate_bands.get(band)})

            lst_df_collection_yr.append(df_band_year)
        df_col_year = pd.concat(lst_df_collection_yr, axis=1)
        # drop duplicated index if applicable
        df_col_year = df_col_year[~df_col_year.index.duplicated(keep='first')]
        dict_collection.update({f'{collection_name}_{str(year)}': df_col_year})

    return dict_collection


def json_to_pandas(data, bands, year_start, 
                year_end, dict_collection,
                collection_name):

    # Only consider one field in the output now
    TS = pd.DataFrame.from_records((data.get(list(data.keys())[0]))
                                    , index=[0]).T
    TS.columns = bands

    for year in np.arange(int(year_start), 
                          int(year_end)+1,1):
        # per year a seperate frame should be created
        lst_df_collection_yr = []
        
        # only clip to the year as cropsar is already interpolated 
        # based on temporal surrounding info
        if 'CROPSAR' in collection_name:
            collection_year = TS[f'{str(year)}-01-01':f'{str(year)}-12-31'] 
        
        # take 6 months around to allow later on interpolation
        else:
            collection_year = TS[f'{str(year-1)}-07-01':f'{str(year+1)}-07-01'] 
        
        
        collection_year.columns = bands
        lst_df_collection_yr.append(collection_year)
    
        df_collection_year = pd.concat(lst_df_collection_yr, axis=1)
        dict_collection.update({f'{collection_name}_{str(year)}': df_collection_year})

    return dict_collection