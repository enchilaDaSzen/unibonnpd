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
import cartopy.crs as ccrs
import wradlib as wrl
from wradlib.dp import process_raw_phidp_vulpiani as kdpvpi
# from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
import twpext as tpx


# =============================================================================
# Define working directory and list files
# =============================================================================
START_TIME = dt.datetime(2021, 7, 13, 0, 0)
# START_TIME = dt.datetime(2019, 1, 15, 6, 50)  #errors reading file
STOP_TIME = START_TIME+dt.timedelta(hours=72)

RSITE = 'JuXPol'
SCAN_ELEVS = {'Vert': 'sweep_0', 'el_280': 'sweep_0', 'el_180': 'sweep_1',
              'el_140': 'sweep_2', 'el_110': 'sweep_3', 'el_82': 'sweep_4',
              'el_60': 'sweep_5', 'el_45': 'sweep_6', 'el_31': 'sweep_7',
              'el_17': 'sweep_8', 'el_06': 'sweep_9'}
SCAN_ELEV = 'Vert'
SCAN_ELEV = 'el_17'

# PDIR = '/media/safe/bonn_postdoc/rd_data/'
CLM_DIR = '/home/dsanchez/sciebo_dsr/codes/github/unibonnpd/qc/xpol_clm/'
MFS_DIR = '/home/dsanchez/sciebo_dsr/codes/github/unibonnpd/qc/xpol_mfs/'

LPFILES = tpx.get_listfilesxpol(RSITE, SCAN_ELEV, START_TIME, STOP_TIME,
                                # parent_dir=None
                                )

# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
N = 495
PLOT_METHODS = True

rdata = tpx.Rad_scan(LPFILES[N], f'{RSITE}')
# rdata.ppi_xpol(scan_elev=SCAN_ELEVS.get(SCAN_ELEV))
rdata.ppi_xpolwrl1(scan_elev=1.7)
rdata.vars['PhiDP [deg]'] *= -1
if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                       rdata.vars)

# %%
# =============================================================================
# rhoHV calibration
# =============================================================================
rcrho = tpx.rhoHV_Noise_Bias(rdata)
rcrho.iterate_radcst(rdata.georef, rdata.params, rdata.vars,
                      plot_method=PLOT_METHODS, data2correct=rdata.vars)
RCDB = rcrho.rhohv_corrs['radar constant [dB]']
rdata.params['radar constant [dB]'] = RCDB

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rcrho.vars,
                                    var2plot='rhoHV [-]')

# %%
# =============================================================================
# PhiDP unfolding
# =============================================================================
rcrho.vars['PhiDP [deg]'] = wrl.dp.unfold_phi(rdata.vars['PhiDP [deg]'],
                                               rcrho.vars['rhoHV [-]'],
                                               width=5,
                                               copy=True)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rdata.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rcrho.vars,
                                    var2plot='PhiDP [deg]')
# %%
# =============================================================================
# Noise suppression
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rcrho.vars, min_snr=2,
                       data2correct=rcrho.vars,
                       plot_method=PLOT_METHODS
                       )
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars)

# %%
# =============================================================================
# Clutter identification and removal
# =============================================================================
# tp.datavis.rad_display.plot_mfs(MFS_DIR)
rnme = tp.eclass.nme.NME_ID(rsnr)
rnme.clutter_id(rdata.georef, rdata.params, rsnr.vars, binary_class=207-64,
                 min_snr=rsnr.min_snr, path_mfs=MFS_DIR,
                 # clmap=np.loadtxt(CLM_DIR+f'{RSITE.lower()}_cluttermap_el0.dat'),
                 data2correct=rsnr.vars, plot_method=PLOT_METHODS)

# %%
# ============================================================================
# Melting layer detection using polarimetric profiles
# ============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 3.5
rmlyr.ml_thickness = 0.5
rmlyr.ml_bottom = rmlyr.ml_top-rmlyr.ml_thickness

# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rczdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rczdr.offset_correction(rnme.vars['ZDR [dB]'],
                         zdr_offset=1.11, data2correct=rnme.vars)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars,
                                    var2plot='ZDR [dB]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rczdr.vars,
                                    var2plot='ZDR [dB]')

# %%
# =============================================================================
# Attenuation Correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rattc.zh_correction(rdata.georef, rdata.params, rczdr.vars,
                     rnme.nme_classif['classif'], mlyr=rmlyr,
                     attc_method='ABRI', pdp_pxavr_azm=5, pdp_dmin=1,
                     pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                     coeff_alpha=[0.1, 0.4, 0.38], coeff_b=[0.76, 0.84, 0.8],
                     coeff_a=[9.781e-5, 1.749e-4, 1.367e-4],
                     plot_method=PLOT_METHODS
                     )

rattc.zdr_correction(rdata.georef, rdata.params, rczdr.vars, rattc.vars,
                      rnme.nme_classif['classif'], mlyr=rmlyr, rhv_thld=0.99,
                      minbins=15, mov_avrgf_len=15, p2avrf=5, method='linear',
                      beta_alpha_ratio=0.139, plot_method=PLOT_METHODS,
                      params={'ZH_lower_lim': 10, 'ZH_upper_lim': 55,
                              'zdr_max': 2.4, 'a1': 0.0528, 'b1': 0.511})

# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# KDP Vulpiani
rkdpv = {}
kdp_vulp = kdpvpi(rattc.vars['PhiDP* [deg]'], winlen=15,
                  dr=rdata.params['gateres [m]']/1000, copy=True)
rkdpv['PhiDP [deg]'] = kdp_vulp[0]
rkdpv['KDP [deg/km]'] = kdp_vulp[1]
rkdpv['PhiDP [deg]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan
rkdpv['KDP [deg/km]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
                                    var2plot='KDP [deg/km]',
                                    vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rkdpv,
                                    var2plot='KDP [deg/km]',
                                    vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

# %%
# =============================================================================
# Rainfall estimation
# =============================================================================
zhr = rattc.vars['ZH [dBZ]']
kdpr = rattc.vars['KDP [deg/km]']
kdpr = rkdpv['KDP [deg/km]']
z_thld = 40

rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe.z_to_r(zhr, a=72, b=2.14, mlyr=rmlyr,
             beam_height=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], a=45.5, b=0.83, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.z_ah_to_r(zhr, rattc.vars['AH [dB/km]'], a1=72,
                b1=2.14, a2=45.5, b2=0.83, z_thld=z_thld, mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(rkdpv['KDP [deg/km]'], a=16.9, b=0.801, mlyr=rmlyr,
               beam_height=rdata.georef['beam_height [km]'])
rqpe.z_kdp_to_r(zhr, rkdpv['KDP [deg/km]'], a1=72,
                 b1=2.14, a2=16.9, b2=0.801, z_thld=z_thld, mlyr=rmlyr,
                 beam_height=rdata.georef['beam_height [km]'])

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                    # rqpe.r_z,
                                    # rqpe.r_z_ah,
                                    rqpe.r_z_kdp,
                                    # var2plot='Rainfall [mm/h]',
                                    # var2plot='PhiDP [deg]',
                                    # proj_suffix='utm',
                                    # data_proj=ccrs.UTM(zone=32),
                                    data_proj=ccrs.PlateCarree(),
                                    proj_suffix='wgs84',
                                    cpy_feats={'status': True},
                                    xlims=[4.3, 9.2], ylims=[52.75, 48.75],
                                    # NRW
                                    # xlims=[4.5, 16.5],ylims=[55.5, 46.5]
                                    # DEU
                                    )
# %%
# =============================================================================
# Creates a new radar object
# =============================================================================
rdatap = copy.deepcopy(rdata)
rdatap.vars['rhoHV [-]'] = rczdr.vars['rhoHV [-]']
rdatap.vars['ZH [dBZ]'] = rattc.vars['ZH [dBZ]']
rdatap.vars['ZDR [dB]'] = rattc.vars['ZDR [dB]']
rdatap.vars['V [m/s]'] = rnme.vars['V [m/s]']
rdatap.vars['PhiDP [deg]'] = rattc.vars['PhiDP [deg]']
rdatap.vars['KDP [deg/km]'] = kdpr
# rdatap.vars['Rainfall [mm/h]'] = rqpe.r_z_ah['Rainfall [mm/h]']

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                       rdatap.vars)

# %%
# if PLOT_METHODS:
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rdatap.vars,
                                    # var2plot='KDP [deg/km]',
                                    mlyr=rmlyr)
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
