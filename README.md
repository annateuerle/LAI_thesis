# LAI_thesis

Leaf Area index thesis of Anna Teuerle.

Because research should be reproducible I publish source code of my thesis here.

Thesis
======

netcdf.py
---------

The code does two things. First, it plots time series of climatic variable for a specific location. Secondly the same climatic variable at the global map scale. It works with monthly time series.

The file has to have this structure: nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{nc_var}.dat.nc','r')

You have to fill in three variables:
startyear: start year of your data 
endyear: end year of your data
nc_variable: shortcut for the climatic variable (pre/vap/tmp/pet)

run
