'''
DISCLAIMER For internal use by EEA only"
'''
import pandas as pd
import os
import sys
sys.path.insert(1,r'/data/sigma/05_PROGRAMS/roelpy/dmp_operational')
import DMPsub
from scipy import stats
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf

flux_csv = r'/data/sigma/EEA_DMP/Data/flux_site_info_v0.2_CLC.csv'
df_flux = pd.read_csv(flux_csv,index_col=0)

dir_csv_in = r'/data/sigma/EEA_DMP/output/sites/csv'

start_date = '2000-02-18'
end_date = '2021-12-28'#remark we do not have callibration data from ICOS yet for 2022 so stop in end 2021

#define if you want to callibrate GPP based on fapar tower or all pixels around tower
tower_pixel="tower"#if you want only tower set this value to "tower" all pixels around tower set to "pixels"

#load all the data at site level to calculate GPP for each site
fp_fapar = os.path.join(dir_csv_in,f'fapar_{tower_pixel}.csv')
fp_tmin = os.path.join(dir_csv_in,r'tmin_s01.csv')
fp_tmax = os.path.join(dir_csv_in,r'tmax_s01.csv')
fp_rad = os.path.join(dir_csv_in,r'rad_s01.csv')

df_fapar = pd.read_csv(fp_fapar,index_col='time',parse_dates=True)
df_tmin = pd.read_csv(fp_tmin,index_col=0,parse_dates=True)
df_tmax = pd.read_csv(fp_tmax,index_col=0,parse_dates=True)
df_rad = pd.read_csv(fp_rad,index_col=0,parse_dates=True)

csv_in = r'/data/sigma/EEA_DMP/Data/co2_monthly.csv'
df_csv = pd.read_csv(csv_in)
date_list = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')

CO2_array = []
for date in date_list:
    print(date)
    yyyy = int(date[0:4])
    mm = int(date[5:7])
    f_CO2conc = df_csv[(df_csv['year'] == int(yyyy)) & (df_csv['month'] == int(mm))]['co2'].values[0]
    print(f_CO2conc)
    CO2_array.append(f_CO2conc)

df_CO2=pd.DataFrame(CO2_array)
df_CO2.index=date_list

values_dict = {'01':'01','02':'01','03':'01','04':'01','05':'01','06':'01','07':'01','08':'01','09':'01','10':'01',
               '11':'11','12':'11','13':'11','14':'11','15':'11','16':'11','17':'11','18':'11','19':'11','20':'11',
               '21':'21','22':'21','23':'21','24':'21','25':'21','26':'21','27':'21','28':'21','29':'21','30':'21','31':'21'}

lst_site = []
#calculate GPP at 10 level with f_RUE set to 1
for siteid in df_fapar:
    df_site_out = pd.DataFrame()
    df_site_out_s10 = pd.DataFrame()

    df_site_fapar = df_fapar[siteid][start_date:end_date]
    df_site_tmin = df_tmin[siteid][start_date:end_date]
    df_site_tmax = df_tmax[siteid][start_date:end_date]
    df_site_rad = df_rad[siteid][start_date:end_date]
    df_site_CO2=df_CO2[start_date:end_date]

    df_dekad = df_site_fapar.index.strftime('%Y%m') +df_site_fapar.index.strftime('%d').map(values_dict)
    df_site_fapar.index = df_dekad
    df_site_fapar_s10 = df_site_fapar.groupby(df_site_fapar.index).mean()

    df_site_tmin.index = df_dekad
    df_site_tmin_s10 = df_site_tmin.groupby(df_site_tmin.index).mean()

    df_site_tmax.index = df_dekad
    df_site_tmax_s10 = df_site_tmax.groupby(df_site_tmax.index).mean()

    df_site_rad.index = df_dekad
    df_site_rad_s10 = df_site_rad.groupby(df_site_rad.index).mean()

    df_site_CO2.index = df_dekad
    df_site_CO2_s10 = df_site_CO2.groupby(df_site_CO2.index).mean()

    arr_fapar = df_site_fapar.to_numpy().astype('float64')
    arr_tmin = df_site_tmin.to_numpy().astype('float64')
    arr_tmax = df_site_tmax.to_numpy().astype('float64')
    arr_rad = df_site_rad.to_numpy().astype('float64')
    arr_CO2= df_site_CO2[0].to_numpy().astype('float64')
    arr_tmin[arr_tmin == 65535] = np.nan
    arr_tmax[arr_tmax == 65535] = np.nan
    arr_rad[arr_rad == 0] = np.nan
    arr_tmin_rescaled = (arr_tmin / 100) - 273.15
    arr_tmax_rescaled = (arr_tmax / 100) - 273.15
    arr_rad_rescaled = arr_rad / 1000

    arr_gppmax = DMPsub.calcDMPmax(arr_tmin_rescaled,arr_tmax_rescaled,arr_rad_rescaled,f_CO2conc=arr_CO2,f_RUE=1,unit=2,GDMPorDMP='GDMP',f_DMPfraction=None)
    arr_gpp = arr_gppmax * arr_fapar

    df_site_out['tmin_degc'] = arr_tmin_rescaled
    df_site_out['tmax_degc'] = arr_tmax_rescaled
    df_site_out['rad_kj'] = arr_rad_rescaled
    df_site_out['fapar'] = arr_fapar
    df_site_out['gppmax'] = arr_gppmax
    df_site_out['gpp'] = arr_gpp
    df_site_out.index = pd.to_datetime(df_site_fapar.index)

    df_site_out_copy = df_site_out.copy()
    df_site_out_copy.index = df_dekad
    df_site_out_copy_s10 = df_site_out_copy.groupby(df_site_out_copy.index).mean()
    df_site_out_copy_s10.index = pd.to_datetime(df_site_out_copy_s10.index)

    arr_fapar_s10 = df_site_fapar_s10.to_numpy().astype('float64')
    arr_tmin_s10 = df_site_tmin_s10.to_numpy().astype('float64')
    arr_tmax_s10 = df_site_tmax_s10.to_numpy().astype('float64')
    arr_rad_s10 = df_site_rad_s10.to_numpy().astype('float64')
    arr_CO2_s10 = df_site_CO2_s10[0].to_numpy().astype('float64')
    arr_tmin_s10[arr_tmin_s10 == 65535] = np.nan
    arr_tmax_s10[arr_tmax_s10 == 65535] = np.nan
    arr_rad_s10[arr_rad_s10 == 0] = np.nan
    arr_tmin_rescaled_s10 = (arr_tmin_s10 / 100) - 273.15
    arr_tmax_rescaled_s10 = (arr_tmax_s10 / 100) - 273.15
    arr_rad_rescaled_s10 = arr_rad_s10 / 1000
    arr_gppmax_s10 = DMPsub.calcDMPmax(arr_tmin_rescaled_s10, arr_tmax_rescaled_s10, arr_rad_rescaled_s10, f_CO2conc=arr_CO2_s10,
                                       f_RUE=1, unit=2, GDMPorDMP='GDMP', f_DMPfraction=None)
    arr_gpp_s10 = arr_gppmax_s10 * arr_fapar_s10

    df_site_out_s10['tmin_degc'] = arr_tmin_rescaled_s10
    df_site_out_s10['tmax_degc'] = arr_tmax_rescaled_s10
    df_site_out_s10['rad_kj'] = arr_rad_rescaled_s10
    df_site_out_s10['fapar'] = arr_fapar_s10
    df_site_out_s10['gppmax'] = arr_gppmax_s10
    df_site_out_s10['gpp'] = arr_gpp_s10
    df_site_out_s10.index = pd.to_datetime(df_site_fapar_s10.index)
    df_site_out_s10['siteid'] = siteid
    lst_site.append(df_site_out_s10)

df_all = pd.concat(lst_site)

#prepare icos gpp dataset
csv_icos_in = r'/data/sigma/EEA_DMP/Data/ICOS/csv/ICOS_GPP_S10_cleaned.csv'#
df_icos = pd.read_csv(csv_icos_in)
df_icos.rename(columns={'level_0': 'date_icos', 'siteid': 'siteid','GPP':'GPP_ICOS'}, inplace=True)
df_icos['GPP_ICOS'].mask((df_icos['GPP_ICOS']<= 0), np.nan, inplace=True)#remove zero and negative values

df_all['date_MODIS']=df_all.index.strftime('%Y-%m-%d')
df_all.rename(columns={'gpp':'GPP_MODIS'}, inplace=True)

df_merge = pd.merge(df_icos, df_all,  how='left', left_on=['siteid','date_icos'], right_on = ['siteid','date_MODIS'],suffixes=('_ICOS', '_MODIS'))
df_merge['GPP_MODIS'].mask((df_merge['GPP_MODIS']<= 0), np.nan, inplace=True)#remove zero and negative values

df_merge.index = pd.to_datetime(df_merge['date_icos'])#
df_merge.dropna(subset=['GPP_ICOS','GPP_MODIS'], inplace=True)#

df_merge= pd.merge(df_merge, df_flux,  how='left', left_on=['siteid'], right_on = df_flux.index)
df_merge.dropna(subset=['igbp'], inplace=True)#(20759, 10)

#for the calibration of the f_RUE parameter in the dmp model we log transform the data and make a mixed model with siteid grouped per biome as random intercept
df_merge["GPP_ICOS_log"]= np.log(df_merge["GPP_ICOS"])
df_merge["GPP_MODIS_log"]= np.log(df_merge["GPP_MODIS"])
vc = {'siteid': '0 + C(siteid)'}
md = smf.mixedlm("GPP_ICOS_log~ GPP_MODIS_log", df_merge, vc_formula=vc, groups="igbp", re_formula="1")
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())
f_RUE_cal_MM=np.exp( mdf.params[0])#get the intercept of the mixed model, this is the log(f_RUE) take the exp to get f_RUE
print(f_RUE_cal_MM)

#make a grouping variable siteid+year to split the dataset in cal val
df_merge.index = pd.to_datetime(df_merge["date_icos"])
df_merge['year']=df_merge.index.year
df_merge['group']=df_merge["siteid"].astype(str) + df_merge["year"].astype(str)
df_merge['group'].value_counts()
df_merge.groupby(["siteid","year"]).count()['date_icos']#.to_csv(f'S:\EEA_DMP\output\sites\csv\icos_dataset_groups.csv')

#splitting the dataset in calibration and validation
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(n_splits=1,test_size=.30,  random_state = 10)

split = splitter.split(df_merge, groups=df_merge['group'])
cal_inds, val_inds = next(split)
cal = df_merge.iloc[cal_inds]
val = df_merge.iloc[val_inds]

cal.to_csv(f'S:\EEA_DMP\output\sites\csv\cal_dataset_{tower_pixel}.csv')
val.to_csv(f'S:\EEA_DMP\output\sites\csv\\val_dataset_{tower_pixel}.csv')

cal['group'].value_counts().to_csv(f'S:\EEA_DMP\output\sites\csv\cal_countobservations_{tower_pixel}.csv')
val['group'].value_counts().to_csv(f'S:\EEA_DMP\output\sites\csv\\val_countobservations_{tower_pixel}.csv')

#calibration for f_RUE from 2.30 to 3.0 with steps of 0.01 bias, precision and accuracy is evaluated on calibration dataset
df_calibration=[]
for i in np.arange(2.30, 3.0, 0.01):
    f_RUE = i
    bias= np.mean(cal['GPP_ICOS']-cal['GPP_MODIS']*f_RUE)
    accuracy = np.sqrt(np.mean((cal['GPP_ICOS']- (cal['GPP_MODIS']*f_RUE)) ** 2))
    precission =accuracy**2- bias** 2
    df_calibration.append((f_RUE,
                           bias,
                           accuracy,
                           precission))
cols = ['f_RUE', 'bias', 'accuracy', 'precission']
calibration_result = pd.DataFrame(df_calibration, columns=cols)
sort_b=calibration_result.sort_values(by=['bias'],ascending=True, ignore_index=True)
sort_a=calibration_result.sort_values(by=['accuracy'],ascending=True, ignore_index=True)
sort_p=calibration_result.sort_values(by=['precission'],ascending=True, ignore_index=True)
f_RUE_cal=(sort_b["f_RUE"][0]+sort_a["f_RUE"][0]+sort_p["f_RUE"][0])/3
print(f_RUE_cal)#

#calculate the GPP_MODIS with the calibrated f_RUE_cal value
val['GPP_MODIS_cal']=val['GPP_MODIS']*f_RUE_cal

dir_out = f'/data/sigma/EEA_DMP/output/sites/validation/per_site_cal_val_{tower_pixel}'
#validation
#1.1 validation per site cal val strategy
df_validation=[]
for siteid in val['siteid'].unique():
    fp_out = os.path.join(dir_out,siteid+'.png')
    df_site = val[val['siteid'] == siteid].dropna(subset=['GPP_ICOS','GPP_MODIS_cal'])#drop rows with no observaiton in ICOS or GPP_MODIS
    if not df_site.empty:
        gradient, intercept, r_value, p_value, std_err = stats.linregress(df_site['GPP_ICOS'], df_site['GPP_MODIS_cal'])
        rmsevalue = np.sqrt(np.mean((df_site['GPP_MODIS_cal'] - df_site['GPP_ICOS']) ** 2))
        rmsevalue_round = '%.2f' % (round(abs(rmsevalue), 2))
        rsquared = '%.2f' % (round(r_value ** 2, 2))

        bias = np.mean(df_site['GPP_ICOS'] - df_site['GPP_MODIS_cal'] )
        accuracy = np.sqrt(np.mean((df_site['GPP_ICOS'] - (df_site['GPP_MODIS_cal'])) ** 2))
        precission = accuracy ** 2 - bias ** 2
        df_validation.append((siteid,
                              bias,
                              accuracy,
                              precission))

        figs, axs = plt.subplots(2, 1, figsize=(35, 14))
        df_site['GPP_ICOS'].plot(ax=axs[0], label='GPP_ICOS', color='green',marker='*',style="o")
        df_site['GPP_MODIS_cal'].plot(ax=axs[0], label='GPP_MODIS_cal', color='black',marker='*',style="o")

        df_site['fapar'].plot(ax=axs[1], label='fAPAR (good and marginal quality)', color='darkred',style="o")
        axs[1].set_ylabel('fAPAR [-]')
        ax2 = axs[1].twinx()
        df_site['gppmax'].plot(ax=ax2, label='GPPmax', color='lime',style="o")

        axs[0].set_ylabel('GPP [gC/m2/day]')
        axs[0].legend(loc='upper left', prop={'size': 9})
        ax2.set_ylabel('GPPmax [gC/m2/day]')
        ax2.legend(loc='upper right', prop={'size': 9})
        axs[1].set_ylabel('fAPAR [-]')
        axs[1].legend(loc='upper left', prop={'size': 9})

        axs[0].set_title('Site: ' + siteid + ' - ' + str(df_flux.loc[siteid]['igbp'])+ str(df_flux.loc[siteid]['CLC_CLASS'])+ '  - ICOS vs GPP MODIS '+'R²=' +str(rsquared)+' RMSE='+str(rmsevalue_round))

        plt.savefig(fp_out)
        plt.close()

cols = ['site_id', 'bias', 'accuracy', 'precission']
validation_result = pd.DataFrame(df_validation, columns=cols)
validation_result.to_csv(os.path.join(dir_out,f'validation_cal_val{tower_pixel}.csv'))

#1.2.1 validation per site MM strategy val dataset
val['GPP_MODISMM_cal']=val['GPP_MODIS']*f_RUE_cal_MM

dir_out = f'/data/sigma/EEA_DMP/output/sites/validation/per_site_cal_val_MM_{tower_pixel}'
#validation
#1 validation per site
df_validation=[]
for siteid in val['siteid'].unique():
    fp_out = os.path.join(dir_out,siteid+'.png')
    df_site = val[val['siteid'] == siteid].dropna(subset=['GPP_ICOS','GPP_MODISMM_cal'])#drop rows with no observaiton in ICOS or GPP_MODIS
    if not df_site.empty:
        gradient, intercept, r_value, p_value, std_err = stats.linregress(df_site['GPP_ICOS'], df_site['GPP_MODISMM_cal'])
        rmsevalue = np.sqrt(np.mean((df_site['GPP_MODISMM_cal'] - df_site['GPP_ICOS']) ** 2))
        rmsevalue_round = '%.2f' % (round(abs(rmsevalue), 2))
        rsquared = '%.2f' % (round(r_value ** 2, 2))

        bias = np.mean(df_site['GPP_ICOS'] - df_site['GPP_MODISMM_cal'] )
        accuracy = np.sqrt(np.mean((df_site['GPP_ICOS'] - (df_site['GPP_MODISMM_cal'])) ** 2))
        precission = accuracy ** 2 - bias ** 2
        df_validation.append((siteid,
                              bias,
                              accuracy,
                              precission))

        figs, axs = plt.subplots(2, 1, figsize=(35, 14))
        df_site['GPP_ICOS'].plot(ax=axs[0], label='GPP_ICOS', color='green',marker='*',style="o")
        df_site['GPP_MODISMM_cal'].plot(ax=axs[0], label='GPP_MODISMM_cal', color='black',marker='*',style="o")

        df_site['fapar'].plot(ax=axs[1], label='fAPAR (good and marginal quality)', color='darkred',style="o")
        axs[1].set_ylabel('fAPAR [-]')
        ax2 = axs[1].twinx()
        df_site['gppmax'].plot(ax=ax2, label='GPPmax', color='lime',style="o")

        axs[0].set_ylabel('GPP [gC/m2/day]')
        axs[0].legend(loc='upper left', prop={'size': 9})
        ax2.set_ylabel('GPPmax [gC/m2/day]')
        ax2.legend(loc='upper right', prop={'size': 9})
        axs[1].set_ylabel('fAPAR [-]')
        axs[1].legend(loc='upper left', prop={'size': 9})

        axs[0].set_title('Site: ' + siteid + ' - ' + str(df_flux.loc[siteid]['igbp'])+ str(df_flux.loc[siteid]['CLC_CLASS'])+ '  - ICOS vs GPP MODIS '+'R²=' +str(rsquared)+' RMSE='+str(rmsevalue_round))

        plt.savefig(fp_out)
        plt.close()

cols = ['site_id', 'bias', 'accuracy', 'precission']
validation_result = pd.DataFrame(df_validation, columns=cols)
validation_result.to_csv(os.path.join(dir_out,f'validation_{tower_pixel}.csv'))

#1.2.2 validation per site MM strategy on whole dataset
df_merge['GPP_MODISMM_cal']=df_merge['GPP_MODIS']*f_RUE_cal_MM

dir_out = f'/data/sigma/EEA_DMP/output/sites/validation/per_site_alldata_MM_{tower_pixel}'
#validation
#1 validation per site
df_validation=[]
for siteid in df_merge['siteid'].unique():
    fp_out = os.path.join(dir_out,siteid+'.png')
    df_site = df_merge[df_merge['siteid'] == siteid].dropna(subset=['GPP_ICOS','GPP_MODISMM_cal'])#drop rows with no observaiton in ICOS or GPP_MODIS
    if not df_site.empty:
        gradient, intercept, r_value, p_value, std_err = stats.linregress(df_site['GPP_ICOS'], df_site['GPP_MODISMM_cal'])
        rmsevalue = np.sqrt(np.mean((df_site['GPP_MODISMM_cal'] - df_site['GPP_ICOS']) ** 2))
        rmsevalue_round = '%.2f' % (round(abs(rmsevalue), 2))
        rsquared = '%.2f' % (round(r_value ** 2, 2))

        bias = np.mean(df_site['GPP_ICOS'] - df_site['GPP_MODISMM_cal'] )
        accuracy = np.sqrt(np.mean((df_site['GPP_ICOS'] - (df_site['GPP_MODISMM_cal'])) ** 2))
        precission = accuracy ** 2 - bias ** 2
        df_validation.append((siteid,
                              bias,
                              accuracy,
                              precission))

        figs, axs = plt.subplots(2, 1, figsize=(35, 14))
        df_site['GPP_ICOS'].plot(ax=axs[0], label='GPP_ICOS', color='green',marker='*',style="o")
        df_site['GPP_MODISMM_cal'].plot(ax=axs[0], label='GPP_MODISMM_cal', color='black',marker='*',style="o")

        df_site['fapar'].plot(ax=axs[1], label='fAPAR (good and marginal quality)', color='darkred',style="o")
        axs[1].set_ylabel('fAPAR [-]')
        ax2 = axs[1].twinx()
        df_site['gppmax'].plot(ax=ax2, label='GPPmax', color='lime',style="o")

        axs[0].set_ylabel('GPP [gC/m2/day]')
        axs[0].legend(loc='upper left', prop={'size': 9})
        ax2.set_ylabel('GPPmax [gC/m2/day]')
        ax2.legend(loc='upper right', prop={'size': 9})
        axs[1].set_ylabel('fAPAR [-]')
        axs[1].legend(loc='upper left', prop={'size': 9})

        axs[0].set_title('Site: ' + siteid + ' - ' + str(df_flux.loc[siteid]['igbp'])+ str(df_flux.loc[siteid]['CLC_CLASS'])+ '  - ICOS vs GPP MODIS '+'R²=' +str(rsquared)+' RMSE='+str(rmsevalue_round))

        plt.savefig(fp_out)
        plt.close()

cols = ['site_id', 'bias', 'accuracy', 'precission']
validation_result = pd.DataFrame(df_validation, columns=cols)
validation_result.to_csv(os.path.join(dir_out,f'validation_{tower_pixel}.csv'))

#2.1 validation all sites together cal val strategy
outfig = f'/data/sigma/EEA_DMP/output/sites/validation/ICOS_vs_VPP_GPP_scatterplot_cal_val_{tower_pixel}.png'

x = val['GPP_ICOS'].values
y = val['GPP_MODIS_cal'].values
gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
rmsevalue = np.sqrt(np.mean((y-x)**2))
print(rmsevalue)

rsquared = '%.3f' % (round(r_value ** 2, 3))
gradient_round = '%.3f' % (round(gradient, 3))
intercept_round = '%.3f' % (round(abs(intercept), 3))
bias='%.3f' % (round((np.average((y-x))), 3))
rmsevalue_round = '%.3f' % (round(abs(rmsevalue), 3))

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=10)
ax.axline(xy1=(0, intercept), slope=gradient)
if (intercept < 0):
    eqtext = 'Y = ' + str(gradient_round) + ' x - ' + str(intercept_round)
else:
    eqtext = 'Y = ' + str(gradient_round) + ' x + ' + str(intercept_round)

rtext = 'R2 = ' + rsquared + ''
rmsetext = 'RMSE = ' + rmsevalue_round
f_rue='f_RUE = '+'%.3f' % (round(f_RUE_cal, 3))
biastext='bias='+bias
fig.text(0.15, 0.80, rtext, verticalalignment='baseline')
fig.text(0.15, 0.75, eqtext, verticalalignment='baseline')
fig.text(0.15, 0.70, rmsetext, verticalalignment='baseline')
fig.text(0.15, 0.65, biastext, verticalalignment='baseline')
fig.text(0.15, 0.60, f_rue, verticalalignment='baseline')

ax.set_xlabel('ICOS GPP [g C/m2/day]')
ax.set_ylabel('GPP_MODIS[g C/m2/day]')
ax.set_xlim([0,30])
ax.set_ylim([0,30])

N = np.arange(30)
ax.plot(N, c='black', linestyle='--')

plt.savefig(outfig)

#2.2.1 validation all sites together MM strategy on val dataset
outfig = f'/data/sigma/EEA_DMP/output/sites/validation/ICOS_vs_VPP_GPP_scatterplot_cal_val_MM_val{tower_pixel}.png'

x = val['GPP_ICOS'].values
y = val['GPP_MODISMM_cal'].values
gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
rmsevalue = np.sqrt(np.mean((y-x)**2))
print(rmsevalue)
rsquared = '%.3f' % (round(r_value ** 2, 3))
gradient_round = '%.3f' % (round(gradient, 3))
intercept_round = '%.3f' % (round(abs(intercept), 3))
bias='%.3f' % (round((np.average((y-x))), 3))
rmsevalue_round = '%.3f' % (round(abs(rmsevalue), 3))

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=10)
ax.axline(xy1=(0, intercept), slope=gradient)
if (intercept < 0):
    eqtext = 'Y = ' + str(gradient_round) + ' x - ' + str(intercept_round)
else:
    eqtext = 'Y = ' + str(gradient_round) + ' x + ' + str(intercept_round)

rtext = 'R2 = ' + rsquared + ''
rmsetext = 'RMSE = ' + rmsevalue_round
f_rue='f_RUE = '+'%.3f' % (round(f_RUE_cal_MM, 3))
biastext='bias='+bias
fig.text(0.15, 0.80, rtext, verticalalignment='baseline')
fig.text(0.15, 0.75, eqtext, verticalalignment='baseline')
fig.text(0.15, 0.70, rmsetext, verticalalignment='baseline')
fig.text(0.15, 0.65, biastext, verticalalignment='baseline')
fig.text(0.15, 0.60, f_rue, verticalalignment='baseline')

ax.set_xlabel('ICOS GPP [g C/m2/day]')
ax.set_ylabel('GPP_MODIS[g C/m2/day]')
ax.set_xlim([0,30])
ax.set_ylim([0,30])

N = np.arange(30)
ax.plot(N, c='black', linestyle='--')

plt.savefig(outfig)

#2.2.2 validation all sites together MM strategy on all dat
outfig = f'/data/sigma/EEA_DMP/output/sites/validation/ICOS_vs_VPP_GPP_scatterplot_alldata_MM_{tower_pixel}.png'

x = df_merge['GPP_ICOS'].values
y = df_merge['GPP_MODISMM_cal'].values
gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
rmsevalue = np.sqrt(np.mean((y-x)**2))
print(rmsevalue)
rsquared = '%.3f' % (round(r_value ** 2, 3))
gradient_round = '%.3f' % (round(gradient, 3))
intercept_round = '%.3f' % (round(abs(intercept), 3))
bias='%.3f' % (round((np.average((y-x))), 3))
rmsevalue_round = '%.3f' % (round(abs(rmsevalue), 3))

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=10)
ax.axline(xy1=(0, intercept), slope=gradient)
if (intercept < 0):
    eqtext = 'Y = ' + str(gradient_round) + ' x - ' + str(intercept_round)
else:
    eqtext = 'Y = ' + str(gradient_round) + ' x + ' + str(intercept_round)

rtext = 'R2 = ' + rsquared + ''
rmsetext = 'RMSE = ' + rmsevalue_round
f_rue='f_RUE = '+'%.3f' % (round(f_RUE_cal_MM, 3))
biastext='bias='+bias
fig.text(0.15, 0.80, rtext, verticalalignment='baseline')
fig.text(0.15, 0.75, eqtext, verticalalignment='baseline')
fig.text(0.15, 0.70, rmsetext, verticalalignment='baseline')
fig.text(0.15, 0.65, biastext, verticalalignment='baseline')
fig.text(0.15, 0.60, f_rue, verticalalignment='baseline')

ax.set_xlabel('ICOS GPP [g C/m2/day]')
ax.set_ylabel('GPP_MODIS[g C/m2/day]')
ax.set_xlim([0,30])
ax.set_ylim([0,30])

N = np.arange(30)
ax.plot(N, c='black', linestyle='--')

plt.savefig(outfig)

#3.1 per biome cal val
df_validation=[]
dir_out = f'/data/sigma/EEA_DMP/output/sites/validation/per_biome_cal_val_{tower_pixel}'
for igbp in val['igbp'].unique():
    fp_out = os.path.join(dir_out,igbp+'.png')
    df_site = val[val['igbp'] == igbp].dropna(subset=['GPP_ICOS','GPP_MODIS_cal'])
    x = df_site['GPP_ICOS'].values
    y = df_site['GPP_MODIS_cal'].values
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    rmsevalue = np.sqrt(np.mean((y - x) ** 2))
    print(rmsevalue)

    rsquared = '%.3f' % (round(r_value ** 2, 3))
    gradient_round = '%.3f' % (round(gradient, 3))
    intercept_round = '%.3f' % (round(abs(intercept), 3))
    bias = '%.3f' % (round((np.average((y - x))), 3))
    rmsevalue_round = '%.3f' % (round(abs(rmsevalue), 3))
    precission= rmsevalue ** 2 - np.average((y - x)) ** 2
    df_validation.append((igbp,
                          bias,
                          rmsevalue_round,
                          precission))
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=10)

    ax.axline(xy1=(0, intercept), slope=gradient)
    if (intercept < 0):
        eqtext = 'Y = ' + str(gradient_round) + ' x - ' + str(intercept_round)
    else:
        eqtext = 'Y = ' + str(gradient_round) + ' x + ' + str(intercept_round)

    rtext = 'R2 = ' + rsquared + ''
    rmsetext = 'RMSE = ' + rmsevalue_round
    f_rue = 'f_RUE = ' + '%.3f' % (round(f_RUE_cal, 3))
    biastext = 'bias=' + bias
    fig.text(0.15, 0.80, rtext, verticalalignment='baseline')
    fig.text(0.15, 0.75, eqtext, verticalalignment='baseline')
    fig.text(0.15, 0.70, rmsetext, verticalalignment='baseline')
    fig.text(0.15, 0.65, biastext, verticalalignment='baseline')
    fig.text(0.15, 0.60, f_rue, verticalalignment='baseline')

    ax.set_xlabel('ICOS GPP [g C/m2/day]')
    ax.set_ylabel('GPP_MODIS[g C/m2/day]')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])

    N = np.arange(30)
    ax.plot(N, c='black', linestyle='--')

    plt.savefig(fp_out)
    plt.close()
cols = ['igbp', 'bias', 'accuracy', 'precission']
validation_result = pd.DataFrame(df_validation, columns=cols)
validation_result.to_csv(os.path.join(dir_out,f'validation_{tower_pixel}.csv'))

#3.2.1 per biome cal val MM on validation dataset only
dir_out = f'/data/sigma/EEA_DMP/output/sites/validation/per_biome_cal_val_MM_{tower_pixel}'
df_validation=[]
for igbp in val['igbp'].unique():
    fp_out = os.path.join(dir_out,igbp+'.png')
    df_site = val[val['igbp'] == igbp].dropna(subset=['GPP_ICOS','GPP_MODISMM_cal'])
    x = df_site['GPP_ICOS'].values
    y = df_site['GPP_MODISMM_cal'].values
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    rmsevalue = np.sqrt(np.mean((y - x) ** 2))
    print(rmsevalue)

    rsquared = '%.3f' % (round(r_value ** 2, 3))
    gradient_round = '%.3f' % (round(gradient, 3))
    intercept_round = '%.3f' % (round(abs(intercept), 3))
    bias = '%.3f' % (round((np.average((y - x))), 3))
    rmsevalue_round = '%.3f' % (round(abs(rmsevalue), 3))
    precission= rmsevalue ** 2 - np.average((y - x)) ** 2
    df_validation.append((igbp,
                          bias,
                          rmsevalue_round,
                          precission))
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=10)

    ax.axline(xy1=(0, intercept), slope=gradient)
    if (intercept < 0):
        eqtext = 'Y = ' + str(gradient_round) + ' x - ' + str(intercept_round)
    else:
        eqtext = 'Y = ' + str(gradient_round) + ' x + ' + str(intercept_round)

    rtext = 'R2 = ' + rsquared + ''
    rmsetext = 'RMSE = ' + rmsevalue_round
    f_rue = 'f_RUE = ' + '%.3f' % (round(f_RUE_cal_MM, 3))
    biastext = 'bias=' + bias
    fig.text(0.15, 0.80, rtext, verticalalignment='baseline')
    fig.text(0.15, 0.75, eqtext, verticalalignment='baseline')
    fig.text(0.15, 0.70, rmsetext, verticalalignment='baseline')
    fig.text(0.15, 0.65, biastext, verticalalignment='baseline')
    fig.text(0.15, 0.60, f_rue, verticalalignment='baseline')

    ax.set_xlabel('ICOS GPP [g C/m2/day]')
    ax.set_ylabel('GPP_MODIS[g C/m2/day]')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])

    N = np.arange(30)
    ax.plot(N, c='black', linestyle='--')

    plt.savefig(fp_out)
    plt.close()

cols = ['igbp', 'bias', 'accuracy', 'precission']
validation_result = pd.DataFrame(df_validation, columns=cols)
validation_result.to_csv(os.path.join(dir_out,f'validation_{tower_pixel}.csv'))

#3.2.2 per biome cal val MM on all data
dir_out = f'/data/sigma/EEA_DMP/output/sites/validation/per_biome_alldata_MM_{tower_pixel}'
df_validation=[]
for igbp in df_merge['igbp'].unique():
    fp_out = os.path.join(dir_out,igbp+'.png')
    df_site = df_merge[df_merge['igbp'] == igbp].dropna(subset=['GPP_ICOS','GPP_MODISMM_cal'])
    x = df_site['GPP_ICOS'].values
    y = df_site['GPP_MODISMM_cal'].values
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    rmsevalue = np.sqrt(np.mean((y - x) ** 2))
    print(rmsevalue)

    rsquared = '%.3f' % (round(r_value ** 2, 3))
    gradient_round = '%.3f' % (round(gradient, 3))
    intercept_round = '%.3f' % (round(abs(intercept), 3))
    bias = '%.3f' % (round((np.average((y - x))), 3))
    rmsevalue_round = '%.3f' % (round(abs(rmsevalue), 3))
    precission= rmsevalue ** 2 - np.average((y - x)) ** 2
    df_validation.append((igbp,
                          bias,
                          rmsevalue_round,
                          precission))
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=10)

    ax.axline(xy1=(0, intercept), slope=gradient)
    if (intercept < 0):
        eqtext = 'Y = ' + str(gradient_round) + ' x - ' + str(intercept_round)
    else:
        eqtext = 'Y = ' + str(gradient_round) + ' x + ' + str(intercept_round)

    rtext = 'R2 = ' + rsquared + ''
    rmsetext = 'RMSE = ' + rmsevalue_round
    f_rue = 'f_RUE = ' + '%.3f' % (round(f_RUE_cal_MM, 3))
    biastext = 'bias=' + bias
    fig.text(0.15, 0.80, rtext, verticalalignment='baseline')
    fig.text(0.15, 0.75, eqtext, verticalalignment='baseline')
    fig.text(0.15, 0.70, rmsetext, verticalalignment='baseline')
    fig.text(0.15, 0.65, biastext, verticalalignment='baseline')
    fig.text(0.15, 0.60, f_rue, verticalalignment='baseline')

    ax.set_xlabel('ICOS GPP [g C/m2/day]')
    ax.set_ylabel('GPP_MODIS[g C/m2/day]')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_title('IGBP Biome: ' + igbp )
    N = np.arange(30)
    ax.plot(N, c='black', linestyle='--')

    plt.savefig(fp_out)
    plt.close()

cols = ['igbp', 'bias', 'accuracy', 'precission']
validation_result = pd.DataFrame(df_validation, columns=cols)
validation_result.to_csv(os.path.join(dir_out,f'validation_{tower_pixel}.csv'))