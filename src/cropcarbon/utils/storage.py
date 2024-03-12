"""
Functions that will contribute in storing the data
"""

# import needed packages

import glob 
import os
from pathlib import Path
import shutil
import numpy as np

def unique_folder_extension(in_folder, out_folder, 
                            extension, overwrite=False,
                            ignore_last_inderscore=True):
    """
    Function that will create for each extension in the input folder, 
    an output folder where this specific file is saved in a separate folder. 
    """
    files = glob.glob(os.path.join(in_folder, f'*{extension}'))

    if ignore_last_inderscore:
        # the last underscore will be removed
        name_folders = [Path(item).stem.split('_')[:-1] for item in files]
    else:
        name_folders = [Path(item).stem for item in files]
    
    name_folders = [item for sublist in name_folders for item in sublist]


    for folder in name_folders:
        out_folder_dataset = os.path.join(out_folder, folder)
        os.makedirs(out_folder_dataset, exist_ok=True)

        # Files already created
        if len(glob.glob(os.path.join(out_folder_dataset, '*'))) > 1 and not overwrite:
            continue
        
        # define which files should be copied
        files_in_folder = glob.glob(os.path.join(in_folder, f'{folder}*'))
        
        # apply the copying
        [shutil.copyfile(item, os.path.join(out_folder_dataset, 
                                            Path(item).name)) 
                                            for item in files_in_folder]
        

def _make_serial(d):
    """
    Function that will ensure that an input object, 
    will be convert to a JSON like format for saving
    """
    if isinstance(d, dict):
        return {_make_serial(k): _make_serial(v)
                for k, v in d.items()}
    elif isinstance(d, list):
        return [_make_serial(i) for i in d]
    elif callable(d):
        return d.__name__
    elif (isinstance(d, float)) and (np.isnan(d)):
        return 'NaN'
    else:
        return d


