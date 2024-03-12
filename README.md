# evoland-cropcarbon
Repository in which all the source code used for the Evoland GPP crop and grassland method development will be stored

"""""
IN-SITU DATA PREPARATION
"""""

Code to pre-process the ICOS data with GPP measurements including cleaning and aggregating the data to 10-daily scale. 
The code can be found under:  scripts/gpp/ref/ICOS_prep_data.py

""""""
DATA EXTRACTION
""""""

Code used to extract all necessary EO and ancillary datasets for the ICOS flux sites.
These datasets are needed to allow satellite-based GPP estimates. The code can be found under scripts/gpp/extractions/get_EO_data_ICOS.py

""""""
LUE calibration
""""""

Code used to re-calibrate the LUE specifically for crop and grassland. The code can be found under scripts/gpp/calibration/LUE_calibr.py


Please note that this is just some example source code to perform these tasks. However, the code needs some local stored datasets to enable processing. 

