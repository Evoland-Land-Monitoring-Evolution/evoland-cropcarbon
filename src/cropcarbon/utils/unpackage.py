""""
Script that will help with unpackage certain data
"""

import os


def unzip_all(folder_dir):
    from zipfile import ZipFile

    subdirs = [item for item in os.listdir(folder_dir) if item.endswith('.zip')]

    for subdir in subdirs:

        if not os.path.exists(os.path.join(folder_dir,
                                           subdir.replace('.zip', ''))):

            with ZipFile(os.path.join(folder_dir,
                                      subdir), 'r') as f:

                # extract in current directory
                f.extractall(os.path.join(folder_dir,
                                          subdir.replace('.zip', '')))
