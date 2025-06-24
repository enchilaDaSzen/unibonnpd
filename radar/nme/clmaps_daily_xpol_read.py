#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:15:40 2024

@author: dsanchez
"""


import datetime as dt
import numpy as np
import towerpy as tp
import pickle
from radar import twpext as tpx
import os
# import traceback
# import wradlib as wrl
# import cartopy.crs as ccrs

# =============================================================================
# Define working directory and list files
# =============================================================================
RADAR_SITE = 'Boxpol'
SCAN_ELEVS = ['n_ppi_280deg', 'n_ppi_180deg', 'n_ppi_140deg', 'n_ppi_110deg',
              'n_ppi_082deg', 'n_ppi_060deg', 'n_ppi_045deg', 'n_ppi_031deg',
              'n_ppi_020deg', 'n_ppi_010deg']
SCAN_ELEV = SCAN_ELEVS[-1]

PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'

START_TIME = dt.datetime(2017, 4, 16, 0, 0)
STOP_TIME =  START_TIME+dt.timedelta(hours=0, minutes=1)

LPFILES = [tpx.get_listfilesxpol(RADAR_SITE, SCAN_ELEV,
                                  START_TIME+dt.timedelta(days=iday),
                                  START_TIME+dt.timedelta(days=iday+1))
            # parent_dir=wdir)
            for iday in range(1)]

# LPFILES[:] = [i for i in LPFILES if len(i) > 0]

RES_DIR = (EWDIR + 'pd_rdres/bxp_clmaps/')


# %%

CMFILES = [RES_DIR + i for i in sorted(os.listdir(RES_DIR))
            if i.endswith('cltmap.tpy') and i.startswith(str(START_TIME.year))]

#2017old (20170403)
# CMFILES = CMFILES[:93]
#2017new (20170416)
CMFILES = CMFILES[93:]

# def qpe_reader(RQPE_FILES, rf2read='r_ah'):
for cnt, cmpf in enumerate(CMFILES):
    with open(cmpf, 'rb') as fpkl:
        if cnt == 0:
            dcmap = pickle.load(fpkl)
            dcmap0 = dcmap['clmap']
            dryd0 = dcmap['ndryscans']
        else:
            print(cmpf)
            dcmap = pickle.load(fpkl)
            dcmap1 = dcmap['clmap']
            dcmap0 = np.nansum((dcmap0, dcmap1), axis=0)
            dryd = dcmap['ndryscans']
            dryd0 = np.nansum((dryd0,dryd))

# %%

clmap = dcmap0 / dryd0

# clmap[clmap < 0.9] = 9e-9

rdata = tpx.Rad_scan(LPFILES[-1][-1], f'{RADAR_SITE}')
rdata.ppi_xpol()

cluttermap_prob = {'Clutter Map [%]': clmap,
                    # 'params': cmdata[0]['params'],
                    # 'georef': cmdata[0]['georef']
                    }

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                cluttermap_prob,
                                var2plot='Clutter Map [%]',
                                vars_bounds={'Clutter Map [%]': [0, 1, 101]},
                                ucmap='tpylsc_useq_bupkyw')

# np.savetxt(RES_DIR+f'boxpol{str(START_TIME.year)}_cluttermap_el0.dat',
#             clmap, fmt='%.7e', delimiter=' ')