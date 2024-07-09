import os
import glob
import pandas as pd
from pathlib import Path


dir = '/data/sigma/Evoland_GPP/data/ICOS/flux/csv/gpp_dd'
dir_meta = '/data/sigma/Evoland_GPP/data/ICOS/flux/meta/flux_site_info_V1_2022_ancillary.csv'
files_png = glob.glob(os.path.join(dir, '**', '*.png'))
files_csv = glob.glob(os.path.join(dir, '**', '*.csv'))


df = pd.read_csv(dir_meta)

sites = list(df.siteid.unique())

files_png_del = [item for item in files_png if Path(item).stem.split('_')[
    0] not in sites]
[os.unlink(item) for item in files_png_del]
files_csv_del = [item for item in files_csv if Path(item).stem.split('_')[
    0] not in sites]

[os.unlink(item) for item in files_csv_del]
