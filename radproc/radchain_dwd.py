#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:13:25 2024

@author: dsanchez
"""

import copy
import datetime as dt
import numpy as np
import towerpy as tp
import wradlib as wrl
# from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
from wradlib.dp import process_raw_phidp_vulpiani as kdpvpi
import cartopy.crs as ccrs
import twpext as tpx

# =============================================================================
# Define working directory and list files
# =============================================================================
START_TIME = dt.datetime(2021, 7, 13, 0, 0)
STOP_TIME = START_TIME+dt.timedelta(hours=72)
RADAR_SITE = 'Essen'  # Essen, Flechtdorf, Neuheilenbach, Offenthal
SCAN_ELEV = 'ppi_vol_0.5'
# SCAN_ELEV = 'ppi_pcp'

LPFILES = tpx.get_listfilesdwd(RADAR_SITE, SCAN_ELEV, START_TIME, STOP_TIME,
                               # parent_dir=WDIR
                               )
PLOT_METHODS = True
# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
N = 495

rdata = tpx.Rad_scan(LPFILES[N], f'{RADAR_SITE}')
rdata.ppi_dwdwrl1(get_rvar='pvars', get_rawvars=True)
# rdata.georefUTM()

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# %%
# =============================================================================
# rhoHV calibration
# =============================================================================
rcrho = tpx.rhoHV_Noise_Bias(rdata)
rcrho.iterate_radcst(rdata.georef, rdata.params, rdata.vars,
                     data2correct=rdata.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rcrho.vars,
                                    var2plot='rhoHV [-]')
# %%
# =============================================================================
# PhiDP unfolding
# =============================================================================
rcrho.vars['PhiDP [deg]'] = wrl.dp.unfold_phi(rdata.vars['PhiDP [deg]'],
                                              rcrho.vars['rhoHV [-]'],
                                              width=5, copy=True)
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
rsnr.signalnoiseratio(rdata.georef, rdata.params, rcrho.vars, min_snr=35,
                      data2correct=rcrho.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars)

# %%
# =============================================================================
# Clutter identification and removal
# =============================================================================
rnme = tp.eclass.nme.NME_ID(rsnr)
rnme.clutter_id(rdata.georef, rdata.params, rsnr.vars, binary_class=159,
                min_snr=rsnr.min_snr, data2correct=rsnr.vars,
                plot_method=PLOT_METHODS)

# %%
# ============================================================================
# Melting layer allocation
# ============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 3.5
rmlyr.ml_thickness = 1
rmlyr.ml_bottom = rmlyr.ml_top - rmlyr.ml_thickness

# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rczdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rczdr.offset_correction(rnme.vars['ZDR [dB]'],
                        zdr_offset=rdata.params['zdr-offset_90deg_pw2'],
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
                    rnme.nme_classif['classif'], mlyr=rmlyr,
                    attc_method='ABRI', pdp_pxavr_azm=1, pdp_dmin=10,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    plot_method=PLOT_METHODS)

rattc.zdr_correction(rdata.georef, rdata.params, rczdr.vars, rattc.vars,
                     rnme.nme_classif['classif'], mlyr=rmlyr, rhv_thld=0.995,
                     minbins=15, mov_avrgf_len=15, p2avrf=5, method='linear',
                     # beta_alpha_ratio=0.139,
                     plot_method=PLOT_METHODS,
                     # params={'ZH_lower_lim': 10, 'ZH_upper_lim': 55,
                     #         'zdr_max': 2.4, 'a1': 0.0528, 'b1': 0.511},
                     )

# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# KDP Vulpiani
rkdpv = {}
winlen = 5
pdp2kadp = rattc.vars['PhiDP* [deg]']
kdp_vulp = kdpvpi(pdp2kadp, winlen=winlen,
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
rz_a, rz_b = 1/0.052, 1/0.57
rkdp_a, rkdp_b = 20.7, 0.72
rah_a, rah_b = 307, 0.92
radp_a, radp_b = 452, 0.98

rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe.z_to_r(zhr, a=rz_a, b=rz_b, mlyr=rmlyr,
            beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(kdpr, a=rkdp_a, b=rkdp_b, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], a=rah_a, b=rah_b, mlyr=rmlyr,
             beam_height=rdata.georef['beam_height [km]'])
rqpe.adp_to_r(rattc.vars['ADP [dB/km]'], a=radp_a, b=radp_b, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.z_ah_to_r(zhr, rattc.vars['AH [dB/km]'], a1=rz_a, b1=rz_b, z_thld=z_thld,
               a2=rah_a, b2=rah_b, mlyr=rmlyr,
               beam_height=rdata.georef['beam_height [km]'])
rqpe.z_kdp_to_r(zhr, kdpr, a1=rz_a, b1=rz_b, z_thld=z_thld,
                a2=rkdp_a, b2=rkdp_b, mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rqpe.r_z,
                                    cpy_feats={'status': True},
                                    # data_proj=ccrs.UTM(zone=32),
                                    # proj_suffix='utm',
                                    data_proj=ccrs.PlateCarree(),
                                    proj_suffix='wgs84',
                                    # xlims=[4.3, 9.2], ylims=[52.75, 48.75],
                                    # NRW
                                    xlims=[4.5, 16.5], ylims=[55.5, 46.5]
                                    # DEU
                                    )

# %%
# =============================================================================
# Create a new radar object
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
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdatap.vars)

# %%
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rdatap.vars)
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
