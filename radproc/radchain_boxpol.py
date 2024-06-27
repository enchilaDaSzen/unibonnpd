#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:55:54 2022

@author: dsanchez
"""

import copy
import datetime as dt
import numpy as np
import towerpy as tp
import wradlib as wrl
from wradlib.dp import process_raw_phidp_vulpiani as kdpvpi
# from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
import cartopy.crs as ccrs
import twpext as tpx


# =============================================================================
# Define working directory and list files
# =============================================================================
START_TIME = dt.datetime(2016, 9, 15, 18, 0)
START_TIME = dt.datetime(2019, 5, 19, 13, 30)
# START_TIME = dt.datetime(2018, 7, 4, 10, 30)
STOP_TIME = START_TIME+dt.timedelta(hours=1)

RADAR_SITE = 'BoXPol'
SCAN_ELEVS = ['n_ppi_280deg', 'n_ppi_180deg', 'n_ppi_140deg', 'n_ppi_110deg',
              'n_ppi_082deg', 'n_ppi_060deg', 'n_ppi_045deg', 'n_ppi_031deg',
              'n_ppi_020deg', 'n_ppi_010deg']
SCAN_ELEV = SCAN_ELEVS[-1]
# SCAN_ELEV = 'n_vertical_scan'

PDIR = '/media/safe/bonn_postdoc/rd_data/'
MFS_DIR = '/home/dsanchez/sciebo_dsr/codes/github/unibonnpd/qc/xpol_mfs/'
CLM_DIR = '/home/dsanchez/sciebo_dsr/codes/github/unibonnpd/qc/xpol_clm/'

LPFILES = tpx.get_listfilesxpol(RADAR_SITE, SCAN_ELEV, START_TIME, STOP_TIME,
                                # parent_dir=PDIR
                                )
# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
N = 3
PLOT_METHODS = False

rdatax = tpx.Rad_scan(LPFILES[N], f'{RADAR_SITE}')
# rdatax.ppi_xpol(rawdata_type=True)
rdatax.ppi_xpolwrl1(rawdata_type=False)
rdatax.georefUTM()
# rdatax.vars['V [m/s]'] *= -1
# rdatax.vars['ZH [dBZ]'] += 1.4

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdatax.georef, rdatax.params,
                                       rdatax.vars)
# %%
# =============================================================================
# rhoHV calibration
# =============================================================================
rcrhox = tpx.rhoHV_Noise_Bias(rdatax)
rcrhox.iterate_radcst(rdatax.georef, rdatax.params, rdatax.vars,
                      data2correct=rdatax.vars, plot_method=PLOT_METHODS)
RCDB = rcrhox.rhohv_corrs['radar constant [dB]']
rdatax.params['radar constant [dB]'] = RCDB

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rcrhox.vars,
                                    var2plot='rhoHV [-]')
# %%
# =============================================================================
# PhiDP unfolding
# =============================================================================
rcrhox.vars['PhiDP [deg]'] = wrl.dp.unfold_phi(rdatax.vars['PhiDP [deg]'],
                                               rcrhox.vars['rhoHV [-]'],
                                               width=5, copy=True)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rdatax.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rcrhox.vars,
                                    var2plot='PhiDP [deg]')
# %%
# =============================================================================
# Noise suppression
# =============================================================================
rsnrx = tp.eclass.snr.SNR_Classif(rdatax)
rsnrx.signalnoiseratio(rdatax.georef, rdatax.params, rcrhox.vars, min_snr=0,
                       data2correct=rcrhox.vars, plot_method=PLOT_METHODS)
rsnrx.vars['ZH [dBZ]'][rsnrx.vars['ZH [dBZ]'] < 5] = np.nan

# %%
# =============================================================================
# Clutter identification and removal
# =============================================================================
rnmex = tp.eclass.nme.NME_ID(rsnrx)
# tp.datavis.rad_display.plot_mfs(MFS_DIR)
rnmex.clutter_id(rdatax.georef, rdatax.params, rsnrx.vars, binary_class=223,
                 min_snr=rsnrx.min_snr, path_mfs=MFS_DIR,
                 clmap=np.loadtxt(CLM_DIR+f'{RADAR_SITE.lower()}'
                                  + '_cluttermap_el0.dat'),
                 data2correct=rsnrx.vars, plot_method=PLOT_METHODS)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rnmex.vars)
# %%
# ============================================================================
# Melting layer allocation
# ============================================================================
rmlyrx = tp.ml.mlyr.MeltingLayer(rdatax)
rmlyrx.ml_top = 3.75
rmlyrx.ml_thickness = 1
rmlyrx.ml_bottom = 2.75
rmlyrx.ml_top = 2.4
rmlyrx.ml_thickness = 0.55
rmlyrx.ml_bottom = 1.85
if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdatax.georef, rdatax.params, rnmex.vars,
                                    mlyr=rmlyrx)

# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rczdrx = tp.calib.calib_zdr.ZDR_Calibration(rdatax)
rczdrx.offset_correction(rnmex.vars['ZDR [dB]'],
                         zdr_offset=-.5,
                         data2correct=rnmex.vars)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rczdrx.vars,
                                    var2plot='ZDR [dB]')

# %%
# =============================================================================
# Attenuation Correction
# =============================================================================
rattcx = tp.attc.attc_zhzdr.AttenuationCorrection(rdatax)
rattcx.zh_correction(rdatax.georef, rdatax.params, rczdrx.vars,
                     rnmex.nme_classif['classif'], mlyr=rmlyrx,
                     attc_method='ABRI', pdp_pxavr_azm=3, pdp_dmin=1,
                     pdp_pxavr_rng=round(4000/rdatax.params['gateres [m]']),
                     # pdp_pxavr_rng=15,
                     coeff_alpha=[0.1, 0.4, 0.38], coeff_b=[0.76, 0.84, 0.8],
                     coeff_a=[9.781e-5, 1.749e-4, 1.367e-4],
                     plot_method=PLOT_METHODS)

rattcx.zdr_correction(rdatax.georef, rdatax.params, rczdrx.vars, rattcx.vars,
                      rnmex.nme_classif['classif'], mlyr=rmlyrx, rhv_thld=0.99,
                      minbins=15, mov_avrgf_len=15, p2avrf=5, method='linear',
                      beta_alpha_ratio=0.139, plot_method=PLOT_METHODS,
                      params={'ZH_lower_lim': 10, 'ZH_upper_lim': 55,
                              'zdr_max': 2.4, 'a1': 0.0528, 'b1': 0.511})

# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# KDP Vulpiani
rkdpvx = {}
if rdatax.params['elev_ang [deg]'] > 8:
    pdp2kadp = rattcx.vars['PhiDP [deg]']
    winlen = 5
else:
    pdp2kadp = rattcx.vars['PhiDP [deg]']
    winlen = 15
kdp_vulp = kdpvpi(pdp2kadp, winlen=winlen,
                  dr=rdatax.params['gateres [m]']/1000, copy=True)
rkdpvx['PhiDP [deg]'] = kdp_vulp[0]
rkdpvx['KDP [deg/km]'] = kdp_vulp[1]
rkdpvx['PhiDP [deg]'][np.isnan(rattcx.vars['ZH [dBZ]'])] = np.nan
rkdpvx['KDP [deg/km]'][np.isnan(rattcx.vars['ZH [dBZ]'])] = np.nan

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rattcx.vars,
                                    var2plot='KDP [deg/km]',
                                    # var2plot='PhiDP [deg]',
                                    # xlims=[-50, 20], ylims=[-40, 50],
                                    vars_bounds={'KDP [deg/km]': (-1, 3, 17)}
                                    )

    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rkdpvx,
                                    var2plot='KDP [deg/km]',
                                    # var2plot='PhiDP [deg]',
                                    # xlims=[-50, 20], ylims=[-40, 50],
                                    vars_bounds={'KDP [deg/km]': (-1, 3, 17)}
                                    )

# %%
# =============================================================================
# Rainfall estimation
# =============================================================================
zhr = rattcx.vars['ZH [dBZ]']
kdpr = rattcx.vars['KDP [deg/km]']
kdpr = rkdpvx['KDP [deg/km]']
z_thld = 40

rqpex = tp.qpe.qpe_algs.RadarQPE(rdatax)
rqpex.z_to_r(zhr, a=72, b=2.14, mlyr=rmlyrx,
             beam_height=rdatax.georef['beam_height [km]'])
rqpex.ah_to_r(rattcx.vars['AH [dB/km]'], a=45.5, b=0.83, mlyr=rmlyrx,
              beam_height=rdatax.georef['beam_height [km]'])
rqpex.adp_to_r(rattcx.vars['ADP [dB/km]'], a=53.3, b=0.85, mlyr=rmlyrx,
               beam_height=rdatax.georef['beam_height [km]'])
rqpex.z_ah_to_r(zhr, rattcx.vars['AH [dB/km]'], a1=72,
                b1=2.14, a2=45.5, b2=0.83, z_thld=z_thld, mlyr=rmlyrx,
                beam_height=rdatax.georef['beam_height [km]'])
rqpex.kdp_to_r(kdpr, a=16.9, b=0.801, mlyr=rmlyrx,
               beam_height=rdatax.georef['beam_height [km]'])
rqpex.z_kdp_to_r(zhr, kdpr, a1=72,
                 b1=2.14, a2=16.9, b2=0.801, z_thld=z_thld, mlyr=rmlyrx,
                 beam_height=rdatax.georef['beam_height [km]'])

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params,
                                    # rqpex.r_ah,
                                    rqpex.r_z_ah,
                                    # rqpex.r_z,
                                    # rqpex.r_kdp,
                                    # rqpex.r_z_kdp,
                                    # xlims=[-50, 50], ylims=[-50, 50],
                                    )
# %%
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params,
                                    rqpex.r_z_ah,
                                    # rdatax.vars,
                                    cpy_feats={'status': True},
                                    data_proj=ccrs.PlateCarree(),
                                    proj_suffix='wgs84',
                                    # data_proj=ccrs.UTM(zone=32),
                                    # proj_suffix='utm',
                                    xlims=[4.3, 9.2], ylims=[52.75, 48.75],
                                    # NRW
                                    # xlims=[4.5, 16.5],ylims=[55.5, 46.5]
                                    # DEU
                                    )

# %%
# =============================================================================
# Creates a new radar object
# =============================================================================
rdataxp = copy.deepcopy(rdatax)
rdataxp.vars['rhoHV [-]'] = rczdrx.vars['rhoHV [-]']
rdataxp.vars['ZH [dBZ]'] = rattcx.vars['ZH [dBZ]']
rdataxp.vars['ZDR [dB]'] = rattcx.vars['ZDR [dB]']
rdataxp.vars['V [m/s]'] = rnmex.vars['V [m/s]']
rdataxp.vars['PhiDP [deg]'] = rattcx.vars['PhiDP [deg]']
rdataxp.vars['KDP [deg/km]'] = kdpr
# rdatap.vars['Rainfall [mm/h]'] = rqpe.r_z_ah['Rainfall [mm/h]']

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdatax.georef, rdatax.params,
                                       rdataxp.vars, mlyr=rmlyrx,
                                       # xlims=[-125, 125], ylims=[-125, 125],
                                       vars_bounds={'PhiDP [deg]': [0, 90, 10],
                                                    'KDP [deg/km]': [-1, 3, 17],
                                                    'ZDR [dB]': [-1, 3, 17]}
                                                   )
    tp.datavis.rad_display.plot_ppi(rdatax.georef, rdatax.params, rdataxp.vars,
                                    cpy_feats={'status': True},
                                    data_proj=ccrs.PlateCarree(),
                                    proj_suffix='wgs84',
                                    xlims=[4.3, 9.2], ylims=[52.75, 48.75],
                                    # NRW
                                    # xlims=[4.5, 16.5],ylims=[55.5, 46.5]
                                    # DEU
                                    )


# %%
tp.datavis.rad_interactive.ppi_base(rdatax.georef, rdatax.params, #rdataxp.vars,
                                    rsnrx.vars,
                                    # var2plot='KDP [deg/km]',
                                    var2plot='rhoHV [-]',
                                    mlyr=rmlyrx
                                    )
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
