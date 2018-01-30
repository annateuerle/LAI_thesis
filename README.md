# LAI_thesis

Leaf Area index thesis of Anna Teuerle.

Because research should be reproducible I publish source code of my thesis here.

Thesis
======

plot_basemap_climatic_variable_CRU.py
--------------------------

The code does two things. First, it plots time series of climatic variable for a specific location. Secondly the same climatic variable at the global map scale. It works with monthly time series.

The file has to have this structure: nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{nc_var}.dat.nc','r')

You have to fill in three variables:
1. startyear: start year of your data 
2. endyear: end year of your data
3. nc_variable: shortcut for the climatic variable (pre/vap/tmp/pet)

run
-----------------------------

time_series.py
-----------------------------

The code makes a time series plot of the LAI from specific location. It is based on LAT and LON

You have to fill in three variables:
1. directory of the folder with LAI data (hdf4 files)
2. LAT and LON of the location
3. For hdf modis files to convert lat,lon to x,y in the 1200x1200 grid the origin coordinates of the measurement, we need to provide pixel size, correct projection. The pixelsize and projection is probably already correct but the origin needs to be specified for each file.
