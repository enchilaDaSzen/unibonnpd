#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:55:54 2022

@author: dsanchez
"""

import datetime as dt
import numpy as np
import towerpy as tp
import wradlib as wrl
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
import cartopy.crs as ccrs
from radar import twpext as tpx

# =============================================================================
# Define working directory and list files
# =============================================================================
RSITE = 'aaxpol'

START_TIME = dt.datetime(2023, 8, 6, 0, 0)
START_TIME = dt.datetime(2024, 3, 11, 12, 0)
STOP_TIME = START_TIME+dt.timedelta(hours=24)
# START_TIME = dt.datetime(2023, 3, 8, 1, 0)
# STOP_TIME = START_TIME+dt.timedelta(minutes=30)

SCAN_ELEVS = {'el_450': 'sweep_8', 'el_250': 'sweep_7', 'el_124': 'sweep_6',
              'el_105': 'sweep_5', 'el_086': 'sweep_0', 'el_067': 'sweep_1',
              'el_048': 'sweep_2', 'el_029': 'sweep_3', 'el_010': 'sweep_4'}
SCAN_ELEV = SCAN_ELEVS.get('el_010')

PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/enchiladasz/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'
PROFSDATA = EWDIR + 'pd_rdres/20210714/'
MFS_DIR = (LWDIR + 'codes/github/unibonnpd/qc/xpol_mfs/')
CLM_DIR = (LWDIR + 'codes/github/unibonnpd/qc/xpol_clm/')

LPFILES = tpx.get_listfilesxpol(RSITE, SCAN_ELEV, START_TIME, STOP_TIME,
                                parent_dir=None)

# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
N = 2
PLOT_METHODS = True

rdata = tpx.Rad_scan(LPFILES[N], f'{RSITE}')
rdata.ppi_xpol(scan_elev=SCAN_ELEV)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                       rdata.vars)

# %%
# =============================================================================
# rhoHV calibration
# =============================================================================
rad_cste = [15, 20, .1]
rcrho = tpx.rhoHV_Noise_Bias(rdata)
rcrho.iterate_radcst(rdata.georef, rdata.params, rdata.vars,
                     data2correct=rdata.vars, plot_method=PLOT_METHODS,
                     rad_cst=None)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rdata.vars,
                                    var2plot='rhoHV [-]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rcrho.vars,
                                    var2plot='rhoHV [-]')
# %%
# =============================================================================
# PhiDP unfolding
# =============================================================================
rcrho.vars['PhiDP [deg]'] = wrl.dp.unfold_phi(rdata.vars['PhiDP [deg]'],
                                              rcrho.vars['rhoHV [-]'],
                                              width=35, copy=True)

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
rsnr.signalnoiseratio(rdata.georef, rdata.params, rcrho.vars,
                      min_snr=-rcrho.rhohv_corrs['Noise level [dB]'],
                      data2correct=rcrho.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars)

# %%
# =============================================================================
# Clutter identification and removal
# =============================================================================
rnme = tp.eclass.nme.NME_ID(rsnr)
rnme.clutter_id(rdata.georef, rdata.params, rsnr.vars, binary_class=21,
                min_snr=rsnr.min_snr, path_mfs=MFS_DIR, clmap=None,
                data2correct=rsnr.vars, plot_method=True)
# %%
# ============================================================================
# Melting layer allocation
# ============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 1.5
rmlyr.ml_thickness = 0.75
rmlyr.ml_bottom = rmlyr.ml_top-rmlyr.ml_thickness

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rnme.vars,
                                       mlyr=rmlyr)

# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rczdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)

rczdr.offset_correction(rnme.vars['ZDR [dB]'], zdr_offset=-1.5,
                        data2correct=rnme.vars)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                    rnme.vars, var2plot='ZDR [dB]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rczdr.vars,
                                    var2plot='ZDR [dB]')

# %%
# =============================================================================
# Attenuation Correction
# =============================================================================

rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rattc.zh_correction(rdata.georef, rdata.params, rczdr.vars,
                    rnme.nme_classif['classif [EC]'], mlyr=rmlyr,
                    attc_method='BRI', pdp_pxavr_azm=3, pdp_dmin=1,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    coeff_alpha=[0.1, 0.4, 0.38], coeff_b=[0.76, 0.84, 0.8],
                    coeff_a=[9.781e-5, 1.749e-4, 1.367e-4],
                    plot_method=PLOT_METHODS)

# %%
zhzdr_a = 0.000249173
zhzdr_b = 2.33327
rattc.zdr_correction(rdata.georef, rdata.params, rczdr.vars, rattc.vars,
                     rnme.nme_classif['classif [EC]'], mlyr=rmlyr, descr=True,
                     rhv_thld=0.98, minbins=9, mov_avrgf_len=3, p2avrf=3,
                     coeff_beta=[0.002, 0.04, 0.02], beta_alpha_ratio=0.165,
                     method='exp', rparams={'a1': zhzdr_a, 'b1': zhzdr_b},
                     plot_method=PLOT_METHODS)

# %%
# =============================================================================
# PBBc and ZHAH
# =============================================================================
rzhah = tp.attc.r_att_refl.Attn_Refl_Relation(rdata)
rzhah.ah_zh(rattc.vars, rband='X', zh_lower_lim=20, zh_upper_lim=50, temp=10,
            copy_ofr=True,  # data2correct=rattc.vars,
            plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars)
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rzhah.vars)

# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# KDP Vulpiani
rkdpv = {}
if rdata.params['elev_ang [deg]'] > 8:
    pdp2kadp = rattc.vars['PhiDP [deg]']
    winlen = 5
else:
    pdp2kadp = rattc.vars['PhiDP* [deg]']
    winlen = 15
kdp_vulp = kdpvpi(pdp2kadp, winlen=winlen,
                  dr=rdata.params['gateres [m]']/1000, copy=True)
rkdpv['PhiDP [deg]'] = kdp_vulp[0]
rkdpv['KDP [deg/km]'] = kdp_vulp[1]
rkdpv['PhiDP [deg]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan
rkdpv['KDP [deg/km]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
                                    var2plot='KDP [deg/km]',
                                    # var2plot='PhiDP [deg]',
                                    xlims=[-50, 20], ylims=[-40, 50],
                                    vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rkdpv,
                                    var2plot='KDP [deg/km]',
                                    # var2plot='PhiDP [deg]',
                                    xlims=[-50, 20], ylims=[-40, 50],
                                    vars_bounds={'KDP [deg/km]': (-1, 3, 17)})
# %%
# =============================================================================
# Rainfall estimation
# =============================================================================
zhr = rattc.vars['ZH [dBZ]']
# zhr = zh_ah['ZH [dBZ]']
zdrr = rattc.vars['ZDR [dB]']
kdpr = rattc.vars['KDP [deg/km]']
# kdpr = rkdpv['KDP [deg/km]']
z_thld = 40

rmlyr2 = None

rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe.z_to_r(zhr, a=72, b=2.14, mlyr=rmlyr,
            beam_height=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r(zhr, zdrr, mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], a=45.5, b=0.83, mlyr=rmlyr,
             beam_height=rdata.georef['beam_height [km]'])
rqpe.adp_to_r(rattc.vars['ADP [dB/km]'], a=53.3, b=0.85, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(kdpr, a=16.9, b=0.801, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.z_ah_to_r(zhr, rattc.vars['AH [dB/km]'], a1=72, b1=2.14, a2=45.5, b2=0.83,
               z_thld=z_thld, mlyr=rmlyr2,
               beam_height=rdata.georef['beam_height [km]'])
rqpe.z_kdp_to_r(zhr, kdpr, a1=72, b1=2.14, a2=16.9, b2=0.801, z_thld=z_thld,
                mlyr=rmlyr, beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_zdr_to_r(kdpr, zdrr, a=28.6, b=0.95, c=-1.37, mlyr=rmlyr,
                  beam_height=rdata.georef['beam_height [km]'])

# PLOT_METHODS=True
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                    # rqpe.r_z,
                                    # rqpe.r_z_zdr,
                                    # rqpe.r_ah,
                                    # rqpe.r_adp,
                                    # rqpe.r_kdp,
                                    rqpe.r_z_ah,
                                    # rqpe.r_z_kdp,
                                    # rqpe.r_kdp_zdr,
                                    # xlims=[-50, 50], ylims=[-50, 50],
                                    )

# %%
xlims, ylims = [5.99, 6.5], [51, 50.4]
xlims, ylims = [5., 8], [51.75, 49]

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,  # rdata.vars,
                                rqpe.r_z_ah,
                                # rdata.vars, # rnme.vars, # rqpe.r_ah,
                                # rqpe.r_z,
                                cpy_feats={'status': True, 'tiles': False,
                                           'tiles_source': 'Stamen',
                                           'tiles_style': 'terrain',
                                           'tiles_res': 8, 'alpha_tiles': 1,
                                           'alpha_rad': 1},
                                data_proj=ccrs.UTM(zone=32),
                                proj_suffix='utm',
                                xlims=xlims, ylims=ylims
                                # xlims=[4.3, 9.2], ylims=[52.75, 48.75],  # NRW
                                # xlims=[4.5, 16.5],ylims=[55.5, 46.5] # DEU
                                )
# %%
# =============================================================================
# Creates a new radar object
# =============================================================================
rd_qcatc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rd_qcatc.georef = rdata.georef
rd_qcatc.params = rdata.params
rd_qcatc.vars = dict(rattc.vars)
rd_qcatc.alpha_ah = np.nanmean([np.nanmean(i)
                                for i in rattc.vars['alpha [-]']])
del rd_qcatc.vars['alpha [-]']
del rd_qcatc.vars['beta [-]']
del rd_qcatc.vars['PIA [dB]']
# del rd_qcatc.vars['PhiDP [deg]']
del rd_qcatc.vars['PhiDP* [deg]']
# rd_qcatc.vars['KDP* [deg/km]'] = rkdpv['KDP [deg/km]']
rd_qcatc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']

# %%
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params,
                                    rd_qcatc.vars)
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
