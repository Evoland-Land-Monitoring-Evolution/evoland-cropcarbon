""" set permissions of folder correct"""

import os
import shutil


def read_write_permission(folder):
    """ set permissions of folder correct"""
    os.chmod(folder, 0o777)
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
            shutil.chown(os.path.join(root, d), user='bontek', group='vito')
        for f in files:
            if os.path.exists(os.path.join(root, f)) and not 'journal' in f: # NOQA
                if 'Thumbs' not in f and not os.path.join(root, f).endswith('.xml'): # NOQA
                    os.chmod(os.path.join(root, f), 0o777)
                    shutil.chown(os.path.join(root, f), user='bontek',
                                 group='vito')

                    
