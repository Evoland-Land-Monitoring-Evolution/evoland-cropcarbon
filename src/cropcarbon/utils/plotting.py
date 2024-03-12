"""
Plotting scripts 
"""

import os
import pandas as pd
from loguru import logger as log
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from cropcarbon.utils.timeseries import clean_TS
import numpy as np
from scipy.stats import gaussian_kde



def _get_plot_TS_(basefolder,
                  outfolder,
                  suffix_outname,
                  overwrite=False):
    files_csv = glob.glob(os.path.join(basefolder,
                                       '**', '*.csv'))
    files_csv = [item for item in files_csv if not 'grouped' in item]

    for file in files_csv:
        year_file = Path(file).stem.split('_')[3]
        site = Path(file).stem.split('_')[0]
        outfolder_year = os.path.join(outfolder,
                                      str(year_file))
        os.makedirs(outfolder_year, exist_ok=True)
        outname = f'{site}_{suffix_outname.format(str(year_file))}'

        if os.path.exists(os.path.join(outfolder_year, outname)) and not overwrite:
            continue

        df_TS = pd.read_csv(file, index_col=0, parse_dates=True)
        # plot only the cleaned timeseries
        df_TS = clean_TS(df_TS)
        df_TS_subset = df_TS[f'{str(year_file)}-01-01':f'{str(year_file)}-12-31']

        if 'GPP_NT_VUT_MEAN' in df_TS_subset.columns:
            df_select_NT = df_TS_subset[['GPP_NT_VUT_MEAN']]
            df_select_NT.rename(
                columns={'GPP_NT_VUT_MEAN': 'GPP_NT'}, inplace=True)

        elif 'GPP_NT_CUT_MEAN' in df_TS_subset.columns:
            df_select_NT = df_TS_subset[['GPP_NT_CUT_MEAN']]
            df_select_NT.rename(
                columns={'GPP_NT_CUT_MEAN': 'GPP_NT'}, inplace=True)

        if 'GPP_DT_VUT_MEAN' in df_TS_subset.columns:
            df_select_DT = df_TS_subset[['GPP_DT_VUT_MEAN']]
            df_select_DT.rename(
                columns={'GPP_DT_VUT_MEAN': 'GPP_DT'}, inplace=True)
        elif 'GPP_DT_CUT_MEAN' in df_TS_subset.columns:
            df_select_DT = df_TS_subset[['GPP_DT_CUT_MEAN']]
            df_select_DT.rename(
                columns={'GPP_DT_CUT_MEAN': 'GPP_DT'}, inplace=True)

        figs, ax2 = plt.subplots(1, 1, figsize=(20, 20))
        ax2.plot(df_select_NT.index, df_select_NT.values,
                 color='black', label='GPP_NT', linewidth=5)
        ax2.plot(df_select_DT.index, df_select_DT.values,
                 color='red', label='GPP_DT', linewidth=5)
        ax2.legend(loc='upper left', fontsize=15)
        ax2.set_title(site + f' - {str(year_file)}', fontsize=18)
        ax2.set_ylabel('GPP [gC/m2/day]', fontsize=15)
        ax2.tick_params(axis="x", labelsize=15)
        ax2.tick_params(axis="y", labelsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(outfolder_year,
                    outname))
        plt.close()


def plot_cal_TS(bands, df, outdir):
    """
    Plot the timeseries used to calibrate the GPP model 
    """
    dict_colors_bands = {
        'GPP_target': 'red',
        'FAPAR': 'green',
        'CROPSAR': 'green',
        'ssm': 'blue'
    }

    # check if all bands are in dataset
    bands_av = [item for item in bands if item in list(df.columns)]

    # filter the dataframe on the bands to plot
    df = df[bands_av]
    # drop empty rows
    df = df.dropna(axis=0)

    if len(bands_av) != len(bands):
        band_mis = list(set(bands)- set(bands_av))
        raise ValueError(f'Missing band(s) for plotting {band_mis}')
    
    figs, ax = plt.subplots(len(bands_av),1, figsize=(20, 20))

    for i in range(len(bands_av)):
        band_plot = bands_av[i]
        if band_plot in dict_colors_bands.keys():
            color = dict_colors_bands.get(band_plot)
        else:
            color = 'black'
        
        ax[i].plot(df.index, df[band_plot].values,
                   color=color, label=band_plot, 
                   linewidth=5)
        ax[i].legend(loc='upper left', fontsize=15)
        if 'GPP_target' in band_plot:
            ax[i].set_ylabel('GPP [gC/m2/day]', fontsize=15)
        elif 'ssm' in band_plot:
            ax[i].set_ylabel('Saturation [%]', fontsize=15) 
        else:
            ax[i].set_ylabel(band_plot, fontsize=15)
        ax[i].tick_params(axis="x", labelsize=15)
        ax[i].tick_params(axis="y", labelsize=15)
    
    plt.tight_layout()
    plt.savefig(outdir)
    plt.close()
    


def plot_cal_param(pred: np.array, obs: np.array, 
                   outdir: str, dict_printing_stats: dict,
                   class_optim: str = None):
    """"
    Plotting of the performance of a calibrated paramter

    Params:
    class_optim: Defines for which type of class the optimization was done
    """
    
    # calculate the point density
    obs_pred = np.vstack([obs, pred])
    z = gaussian_kde(obs_pred)(obs_pred)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = obs[idx], pred[idx], z[idx]

    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.scatter(x, y, c=z, s=50)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    ax.set_ylabel('GPP [gC/m2/day] target', fontsize=30)
    ax.set_xlabel('GPP [gC/m2/day] model', fontsize=30)

    ### add also the text to the fig
    lst_text = []
    for stat_type, stat_out in dict_printing_stats.items():
        lst_text.append(f'{stat_type}: {stat_out}')
    textstr = '\n'.join(lst_text)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
    
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", label='1:1 line',
                             color='red')
    if class_optim is not None:
        ax.set_title(f'CAL FOR {class_optim}', fontsize=40)
    plt.tight_layout()
    plt.savefig(outdir)
    plt.close()
