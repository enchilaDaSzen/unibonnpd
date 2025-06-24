#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:37:17 2024

@author: dsanchez
"""

import os
import sys
import datetime as dt
LWDIR = '/home/dsanchez/sciebo_dsr/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
sys.path.append(LWDIR + 'codes/github/unibonnpd/')
from radar import twpext as tpx


# =============================================================================
# Define working directory, time and list files
# =============================================================================
START_TIME = dt.datetime(2020, 6, 13, 0, 0)
START_TIME = dt.datetime(2018, 12, 2, 0, 0)

EVNTD_HRS = 24
STOP_TIME = START_TIME + dt.timedelta(hours=EVNTD_HRS)

EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

# =============================================================================
# Read RG data
# =============================================================================
RG_WDIR = (EWDIR + 'pd_rdres/dwd_rg/')
DWDRG_MDFN = (RG_WDIR + 'RR_Stundenwerte_Beschreibung_Stationen2024.csv')
RG_NCDATA = (EWDIR + f"pd_rdres/dwd_rg/nrw_{START_TIME.strftime('%Y%m%d')}"
             + f"_{STOP_TIME.strftime('%Y%m%d')}_1h_1hac/")
if not os.path.exists(RG_NCDATA):
    # os.mkdir(f'{ddir}')
    print('Dir was created')
    os.makedirs(RG_NCDATA, exist_ok=True)

# =============================================================================
# Init raingauge object
# =============================================================================
rg_data = tpx.RainGauge(RG_WDIR, nwk_opr='DWD')

# =============================================================================
# Read metadata of all DWD rain gauges (location, records, etc)
# =============================================================================
rg_data.get_dwdstn_mdata(DWDRG_MDFN, plot_methods=False)

# =============================================================================
# Get rg locations within radar coverage
# =============================================================================
# rg_data.get_stns_rad(rqpe_acc.georef, rqpe_acc.params, rg_data.dwd_stn_mdata,
#                     # dmax2rad=rqpe_acc.georef['range [m]'][-75]/1000,
#                     dmax2rad=50,
#                     dmax2radbin=1, plot_methods=PLOT_METHODS)

# =============================================================================
# Get rg locations within bounding box
# =============================================================================
# bbox_xlims, bbox_ylims = (6, 9.2), (49.35, 52.32)  # XPOL
# bbox_xlims, bbox_ylims = (5.5, 8.5), (49.7, 52.18)  # XPOLF
bbox_xlims, bbox_ylims = (6, 11.02), (48.55, 52.77)  # DWDXPOL
# bbox_xlims, bbox_ylims = (6, 10.7), (49, 52.6)  # DWDXPOLF

rg_data.get_stns_box(rg_data.dwd_stn_mdata, bbox_xlims=bbox_xlims,
                     bbox_ylims=bbox_ylims, plot_methods=False,
                     surface=None)

# =============================================================================
# Download DWD rg data
# =============================================================================
for hour in range(EVNTD_HRS):
    # start_time = dt.datetime(2018, 9, 23, 0, 0, 0)
    # start_time = dt.datetime(2020, 6, 17, 0, 0, 0)
    start_time1 = START_TIME
    # start_time = dt.datetime(2017, 7, 26, 0, 0)
    start_time1 = start_time1 + dt.timedelta(hours=hour)
    stop_time = start_time1 + dt.timedelta(hours=1)
    # start_time = start_time + datetime.timedelta(hours=hour+1)
    # for station_id in rg_data.stn_near_rad['stations_id']:
    for station_id in rg_data.stn_bbox['station_id']:
        rg_data.get_dwdstn_nc(station_id, start_time1, stop_time,
                              dir_ncdf=RG_NCDATA)

# =============================================================================
# Read DWD rg data
# =============================================================================
# rg_data.get_rgdata(resqpe_accd_params['datetime'], ds_ncdir=RG_NCDATA,
#                    drop_nan=True, drop_thrb=None, ds2read=rg_data.stn_bbox,
#                    ds_tres=dt.timedelta(hours=1),
#                    # dt_bkwd=dt.timedelta(hours=3),
#                    # ds_accum=dt.timedelta(hours=3),
#                    dt_bkwd=dt.timedelta(hours=1),
#                    ds_accum=dt.timedelta(hours=1),
#                    plot_methods=False)