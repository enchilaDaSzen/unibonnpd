#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:13:25 2024

@author: dsanchez
"""

import datetime as dt
import numpy as np
import towerpy as tp
from radar import twpext as tpx
# from towerpy.utils import unit_conversion as tpuc
import wradlib as wrl
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
import cartopy.crs as ccrs
from radar.rparams_dwdxpol import RPARAMS
# import matplotlib as mpl

# =============================================================================
# Define working directory and list files
# =============================================================================
# START_TIME = dt.datetime(2017, 7, 19, 0, 0)  # HNNVR HAIL
# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24hr [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24hr [NO JXP]
# START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24hr
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24 hr [NO JXP]
START_TIME = dt.datetime(2018, 12, 2, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0) # 24 hr [NO JXP]
# START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 14hr [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 0, 0)  # 24 hr []
# START_TIME = dt.datetime(2020, 6, 14, 0, 0)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2021, 5, 25, 0, 0)  # 24 hr
# START_TIME = dt.datetime(2021, 6, 20, 0, 0)  # 24 hr
# START_TIME = dt.datetime(2021, 6, 29, 0, 0)  # 24 hr
START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24 hr
# START_TIME = dt.datetime(2024, 5, 17, 0, 0)  # 24 hr

STOP_TIME = START_TIME+dt.timedelta(hours=24)
# STOP_TIME = START_TIME+dt.timedelta(minutes=8)

# Essen, Flechtdorf, Neuheilenbach, Offenthal, Hannover
RADAR_SITE = 'Essen'
# SCAN_ELEV = 'ppi_vol_25.0'
# SCAN_ELEV = 'ppi_vol_12.0'
# SCAN_ELEV = 'ppi_vol_1.5'
SCAN_ELEV = 'ppi_pcp'

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
PDIR = None
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'
# PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'

RPARAMS = [i for i in RPARAMS if i['site_name'] == RADAR_SITE]

RS_FILES = tpx.get_listfilesdwd(RADAR_SITE, START_TIME, STOP_TIME,
                                scan_elev=SCAN_ELEV, parent_dir=PDIR)
PLOT_METHODS = False
# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
LPFILE = RS_FILES[180]  # UNFOLDING 20180923QVPS Flechtdorf
LPFILE = RS_FILES[212]
# LPFILE = RS_FILES[74]

rdata = tpx.Rad_scan(LPFILE, f'{RADAR_SITE}')
rdata.ppi_dwd(get_rawvars=True)

tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)
if PLOT_METHODS:
    # Plot cone coverage
    tp.datavis.rad_display.plot_cone_coverage(rdata.georef, rdata.params,
                                              rdata.vars,
                                              # var2plot='PhiDP [deg]',
                                              # var2plot='V [m/s]',
                                              # var2plot='rhoHV [-]',
                                              # var2plot='ZDR [dB]',
                                              limh=6,  zlims=[0, 6])

# %%
# =============================================================================
# rhoHV noise-correction
# =============================================================================
if START_TIME.year >= 2021:
    rhohv_theo, noise_lvl = (0.95, 1.), (32, 35, 0.01)  # 12 deg
else:
    rhohv_theo, noise_lvl = (0.95, 1.), (28, 33, 0.01)  # 12 deg
rhohv_theo, noise_lvl = (0.90, 1.1), (36, 42, 0.01)  # precip_scan
# rhohv_theo, noise_lvl = (0.90, 1.1), None  # precip_scan
rcrho = tpx.rhoHV_Noise_Bias(rdata)
rcrho.iterate_radcst(rdata.georef, rdata.params, rdata.vars,
                     rhohv_theo=rhohv_theo, noise_lvl=noise_lvl,
                     data2correct=rdata.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rdata.vars, rcrho.vars,
        var2plot1='rhoHV [-]', var2plot2='rhoHV [-]',
        ucmap_diff='tpylsc_div_dbu_rd', diff_lims=[-0.5, 0.5, .1])

# %%
# =============================================================================
# Noise suppression
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
# print(f"minSNR = {rcrho.rhohv_corrs['Noise level [dB]']:.2f}")
if rdata.params['radar constant [dB]'] <= 0:
    min_snr = -rcrho.rhohv_corrs['Noise level [dB]']
else:
    min_snr = rcrho.rhohv_corrs['Noise level [dB]']
print(f"minSNR = {min_snr:.2f} dB")
rsnr.signalnoiseratio(rdata.georef, rdata.params, rcrho.vars, min_snr=min_snr,
                      data2correct=rcrho.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rsnr.vars)

# %%
# =============================================================================
# PhiDP quality control and processing
# =============================================================================

# rsnr.vars['PhiDP [deg]'] = np.ascontiguousarray(
#     wrl.dp.unfold_phi(rsnr.vars['PhiDP [deg]'],
#                       rsnr.vars['rhoHV [-]'],
#                       width=1, copy=True).astype(np.float64))

ropdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)
ropdp.offsetdetection_ppi(rsnr.vars, preset=None, mode='median')
# ropdp.phidp_offset = 40
print(f'Phi_DP(0) = {np.median(ropdp.phidp_offset):.2f}')
ropdp.offset_correction(rsnr.vars['PhiDP [deg]'],
                        phidp_offset=ropdp.phidp_offset,
                        data2correct=rsnr.vars)
# PLOT_METHODS = True
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='PhiDP [deg]')

ropdp.vars['PhiDP [deg]'] = np.ascontiguousarray(
    wrl.dp.unfold_phi(ropdp.vars['PhiDP [deg]'],
                      ropdp.vars['rhoHV [-]'],
                      width=3, copy=True).astype(np.float64))

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='PhiDP [deg]')

# %%
# =============================================================================
# Clutter identification and removal
# =============================================================================
rdata2 = tpx.Rad_scan(LPFILE, f'{RADAR_SITE}')
# DWD clutter map is not always available, these lines try to read such data
try:
    rdata2.ppi_dwd(get_rvar='cmap')

    cmap = 1 - tp.utils.radutilities.normalisenanvalues(
        rdata2.vars['cmap [0-1]'], np.nanmin(rdata2.vars['cmap [0-1]']),
        np.nanmax(rdata2.vars['cmap [0-1]']))
    cmap = np.nan_to_num(cmap, nan=1e-5)
    bclass = 207
    pass
except Exception:
    cmap = None
    bclass = 207 - 64
    print('No CL Map available')
    pass

rnme = tp.eclass.nme.NME_ID(ropdp)
rnme.lsinterference_filter(rdata.georef, rdata.params, ropdp.vars,
                           rhv_min=0.3, data2correct=ropdp.vars,
                           plot_method=PLOT_METHODS)
rnme.clutter_id(rdata.georef, rdata.params, rnme.vars, binary_class=bclass,
                min_snr=rsnr.min_snr, clmap=cmap, data2correct=rnme.vars,
                plot_method=PLOT_METHODS)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars)
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars)

# %%
# ============================================================================
# Melting layer allocation
# ============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
# rmlyr.ml_top = 4.4
# rmlyr.ml_thickness = 1.97
rmlyr.ml_top = 3.2
rmlyr.ml_bottom = 1.6
rmlyr.ml_thickness = rmlyr.ml_top - rmlyr.ml_bottom

rmlyr.ml_ppidelimitation(rdata.georef, rdata.params, rsnr.vars, 
                         plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rnme.vars,
                                       mlyr=rmlyr)
# %%
# =============================================================================
# ZDR offset correction
# =============================================================================
rozdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rozdr.offset_correction(rnme.vars['ZDR [dB]'], zdr_offset=-0.15,
                        data2correct=rnme.vars)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars,
                                    var2plot='ZDR [dB]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='ZDR [dB]')

# %%
# =============================================================================
# ZH attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)

rattc.attc_phidp_prepro(
    rdata.georef, rdata.params, rozdr.vars, rhohv_min=0.85,
    phidp0_correction=(True if (rattc.site_name == 'Offenthal') else False))

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
                                    var2plot='PhiDP [deg]')

rattc.zh_correction(rdata.georef, rdata.params, rattc.vars,
                    rnme.nme_classif['classif [EC]'], mlyr=rmlyr, pdp_dmin=1,
                    attc_method='ABRI', phidp0=0, pdp_pxavr_azm=3,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    # coeff_a=[1.59e-5, 4.27e-5, 2.49e-05],  # Diederich
                    # coeff_b=[0.73, 0.77, 0.755],  # Diederich
                    # coeff_alpha=[0.04, 0.1, 0.08],
                    coeff_alpha=[0.08, 0.18, 0.11],
                    # coeff_alpha=[0.05, 0.18, 0.11],
                    plot_method=True)

# %%
# =============================================================================
# Partial beam blockage correction
# =============================================================================
temp = 15

rzhah = tp.attc.r_att_refl.Attn_Refl_Relation(rdata)
rzhah.ah_zh(rattc.vars, zh_upper_lim=55, temp=temp, rband='C',
            copy_ofr=True, data2correct=rattc.vars)
rattc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']

mov_avrgf_len = (1, 3)
zh_difnan = np.where(rzhah.vars['diff [dBZ]'] == 0, np.nan,
                     rzhah.vars['diff [dBZ]'])
zhpdiff = np.array([np.nanmedian(i) if ~np.isnan(np.nanmedian(i))
                    else 0 for cnt, i in enumerate(zh_difnan)])
zhpdiff_pad = np.pad(zhpdiff, mov_avrgf_len[1]//2, mode='wrap')
zhplus_maf = np.ma.convolve(
    zhpdiff_pad, np.ones(mov_avrgf_len[1])/mov_avrgf_len[1],
    mode='valid')
rattc.vars['ZH+ [dBZ]'] = np.array(
    [rattc.vars['ZH [dBZ]'][cnt] - i if i == 0
     else rattc.vars['ZH [dBZ]'][cnt] - zhplus_maf[cnt]
     for cnt, i in enumerate(zhpdiff)])

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rattc.vars, rattc.vars,
        var2plot1='ZH [dBZ]', var2plot2='ZH* [dBZ]')

# %%
# =============================================================================
# ZDR Attenuation Correction
# =============================================================================
zhzdr_a = 0.000249173
zhzdr_b = 2.33327
rb_a = 0.39  # Continental
rb_a = 0.14  # Tropical

rattc.zdr_correction(rdata.georef, rdata.params, rozdr.vars, rattc.vars,
                     rnme.nme_classif['classif [EC]'], mlyr=rmlyr, descr=True,
                     coeff_beta=[0.002, 0.07, 0.04], beta_alpha_ratio=rb_a,
                     rhv_thld=0.98, mov_avrgf_len=9, minbins=5, p2avrf=3,
                     attc_method='BRI', zh_zdr_model='exp',
                     rparams={'coeff_a': zhzdr_a, 'coeff_b': zhzdr_b},
                     plot_method=PLOT_METHODS)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(rdata.georef, rdata.params, rozdr.vars,
                                        rattc.vars, var2plot1='ZDR [dB]',
                                        var2plot2='ZDR [dB]',
                                        diff_lims=[-5, 5, .5])

# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# KDP Vulpiani
zh_kdp = 'ZH+ [dBZ]'
rkdpv = {}
kdp_vulp = kdpvpi(rattc.vars['PhiDP [deg]'], winlen=3,
                  dr=rdata.params['gateres [m]']/1000, copy=True)
rkdpv['PhiDP [deg]'] = kdp_vulp[0]
rkdpv['KDP [deg/km]'] = kdp_vulp[1]

# Remove NME
# rattc.vars['KDP* [deg/km]'] = np.where(rnme.nme_classif['classif [EC]'] != 0,
#                                        np.nan, rkdpv['KDP [deg/km]'])
rattc.vars['KDP* [deg/km]'] = np.where(rnme.ls_dsp_class['classif [EC]'] != 0,
                                       np.nan, rkdpv['KDP [deg/km]'])

# Remove negative KDP values in rain region and within ZH threshold
rattc.vars['KDP* [deg/km]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rkdpv['KDP [deg/km]'] < 0) & (rattc.vars[zh_kdp] > 0),
    # 0, rkdpv['KDP [deg/km]'])
    0, rattc.vars['KDP* [deg/km]'])
# Filter KDP by applying thresholds in ZH and rhoHV
rattc.vars['KDP+ [deg/km]'] = np.where(
    (rattc.vars[zh_kdp] > 40) & (rattc.vars[zh_kdp] <= 55)
    & (rozdr.vars['rhoHV [-]'] > 0.95) & (rattc.vars['KDP [deg/km]'] != 0)
    & (~np.isnan(rattc.vars['KDP [deg/km]'])),
    rattc.vars['KDP [deg/km]'], rattc.vars['KDP* [deg/km]'])

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rattc.vars, rattc.vars,
        var2plot1='KDP* [deg/km]', var2plot2='KDP+ [deg/km]',
        diff_lims=[-1, 1, .25], vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

# %%
# =============================================================================
# Rainfall estimation
# =============================================================================
zh_r = 'ZH+ [dBZ]'  # ZH(AH)
kdp_r = 'KDP+ [deg/km]'  # Vulpiani+AH
zdr_r = 'ZDR [dB]'  # ZDR_ATTC
zh_kdpo = 'ZH [dBZ]'  # ZH(AH)
zh_zho = 'ZH+ [dBZ]'  # ZH(AH)
kdp_kdpo = 'KDP [deg/km]'  # KDP(AH)
z_thld = 40

rz_a, rz_b = (1/0.052)**(1/0.57), 1/0.57  # Chen2021
# rz_a, rz_b = (1/0.026)**(1/0.69), 1/0.69  # Chen2023
rkdp_a, rkdp_b = 20.7, 0.72  # Chen2021
# rkdp_a, rkdp_b = 30.6, 0.71  # Chen2023
# rah_a, rah_b = 307, 0.92  # Chen2021
# radp_a, radp_b = 452, 0.98  # Chen2021

# rz_a, rz_b = 175.2118, 1.6  # ZHadaptive
rz_ahail, rz_bhail = (1/0.022)**(1/0.61), 1/0.61  # Chen2023

rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe.adp_to_r(rattc.vars['ADP [dB/km]'], temp=temp, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], temp=temp, mlyr=rmlyr,
             beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(rattc.vars[kdp_r], a=rkdp_a, b=rkdp_b, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_zdr_to_r(rattc.vars[kdp_r], rattc.vars[zdr_r], mlyr=rmlyr,
                  beam_height=rdata.georef['beam_height [km]'])
rqpe.z_to_r(rattc.vars[zh_r], a=rz_a, b=rz_b, mlyr=rmlyr,
            beam_height=rdata.georef['beam_height [km]'])
rqpe.z_ah_to_r(rattc.vars[zh_r], rattc.vars['AH [dB/km]'], rz_a=rz_a,
               rz_b=rz_b, temp=temp, z_thld=z_thld, mlyr=rmlyr,
               beam_height=rdata.georef['beam_height [km]'])
rqpe.z_kdp_to_r(rattc.vars[zh_r], rattc.vars[kdp_r], rz_a=rz_a, rz_b=rz_b,
                rkdp_a=rkdp_a, rkdp_b=rkdp_b, mlyr=rmlyr, z_thld=z_thld,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r(rattc.vars[zh_r], rattc.vars[zdr_r], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
# ZH(R) Adaptive
rzh_fit = tpx.rzh_opt(
    rattc.vars[zh_zho], rqpe.r_ah, rattc.vars['AH [dB/km]'], mlyr=rmlyr,
    pia=rattc.vars['PIA [dB]'], maxpia=10, rz_stv=(rz_a, rz_b),
    plot_method=PLOT_METHODS)
rqpe_opt = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_opt.z_to_r(rattc.vars[zh_r], a=rzh_fit[0], b=rzh_fit[1], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.r_zopt = rqpe_opt.r_z
# RKDP Adaptive
rkdp_fit = tpx.rkdp_opt(rattc.vars[kdp_kdpo], rattc.vars[zh_kdpo], mlyr=rmlyr,
                        rband='C', plot_method=PLOT_METHODS)
rqpe_opt.kdp_to_r(
    rattc.vars[kdp_r], a=rkdp_fit[0], b=rkdp_fit[1],
    mlyr=rmlyr, beam_height=rdata.georef['beam_height [km]'])
rqpe.r_kdpopt = rqpe_opt.r_kdp

qpe_amlb = False
thr_zwsnw = 0
thr_zhail = 55
if qpe_amlb:
    f_rz_ml = 0.6
    f_rz_sp = 2.8
else:
    f_rz_ml = 0.
    f_rz_sp = 0.
# =============================================================================
# Additional factor for the RZ relation is applied to data within the ML
# =============================================================================
rqpe_ml = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_ml.z_to_r(rattc.vars[zh_r], a=rz_a, b=rz_b)
rqpe_ml.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 2)
    & (rattc.vars[zh_r] > thr_zwsnw),
    rqpe_ml.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)
# =============================================================================
# Additional factor for the RZ relation is applied to data above the ML
# =============================================================================
rqpe_sp = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_sp.z_to_r(rattc.vars[zh_r], a=rz_a, b=rz_b)
rqpe_sp.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 3.),
    rqpe_sp.r_z['Rainfall [mm/h]']*f_rz_sp, rqpe_ml.r_z['Rainfall [mm/h]'])
# Correct all other variables
[setattr(rqpe, rp, {(k1): (np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
    getattr(rqpe, rp)['Rainfall [mm/h]'],
    rqpe_sp.r_z['Rainfall [mm/h]']) if 'Rainfall' in k1 else v1)
    for k1, v1 in getattr(rqpe, rp).items()})
    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]
# =============================================================================
# rz_hail is applied to data below the ML with Z > 55 dBZ
# =============================================================================
rqpe_hail = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_hail.z_to_r(rattc.vars[zh_r], a=rz_ahail, b=rz_bhail)
rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rattc.vars[zh_r] >= thr_zhail),
    rqpe_hail.r_z['Rainfall [mm/h]'], 0)
# Correct all other variables
[setattr(rqpe, rp, {(k1): (np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rattc.vars[zh_r] >= thr_zhail), rqpe_hail.r_z['Rainfall [mm/h]'],
    getattr(rqpe, rp)['Rainfall [mm/h]']) if 'Rainfall' in k1 else v1)
    for k1, v1 in getattr(rqpe, rp).items()})
    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]

if PLOT_METHODS:
    rests = [i for i in sorted(dir(rqpe)) if i.startswith('r_kdpopt')
             or i.startswith('r_zopt')]
    for i in rests:
        tp.datavis.rad_display.plot_ppi(
            rdata.georef, rdata.params, getattr(rqpe, i),
            fig_title=f"{rdata.params['elev_ang [deg]']:{2}.{3}} deg. --"
            + f" {rdata.params['datetime']:%Y-%m-%d %H:%M:%S} -- {i}")
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                    # rqpe.r_ah,
                                    # rqpe.r_kdp,
                                    rqpe.r_z,
                                    # rqpe.r_z_ah,
                                    cpy_feats={
                                        'status': True,
                                        # 'tiles': True, 'alpha_rad': 0.25
                                               },
                                    data_proj=ccrs.UTM(zone=32),
                                    proj_suffix='utm',
                                    # var2plot='rhoHV [-]'
                                    # xlims=[4.3, 9.2], ylims=[52.75, 48.75],
                                    # NRW
                                    # xlims=[4.5, 16.5], ylims=[55.5, 46.5]
                                    # DEU
                                    )

# %%
# =============================================================================
# OUTPUT
# =============================================================================
rd_qcatc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rd_qcatc.georef = rdata.georef
rd_qcatc.params = rdata.params
rd_qcatc.vars = dict(rattc.vars)

del rd_qcatc.vars['alpha [-]']
del rd_qcatc.vars['beta [-]']
# del rd_qcatc.vars['PIA [dB]']
# del rd_qcatc.vars['PhiDP [deg]']
del rd_qcatc.vars['PhiDP* [deg]']
del rd_qcatc.vars['ADP [dB/km]']
# del rd_qcatc.vars['AH [dB/km]']
# rd_qcatc.vars['AH [dB/km]'] =
# rd_qcatc.vars['KDP* [deg/km]'] = rkdpv['KDP [deg/km]']
# rd_qcatc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']
rd_qcatc.vars['rhoHV [-]'] = rozdr.vars['rhoHV [-]']
rd_qcatc.vars['Rainfall [mm/h]'] = rqpe.r_zopt['Rainfall [mm/h]']

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rd_qcatc.georef, rd_qcatc.params,
                                       rd_qcatc.vars, mlyr=rmlyr)

# %%
# if PLOT_METHODS:
tp.datavis.rad_interactive.ppi_base(
    rdata.georef, rdata.params,
    # rdata.vars,
    # rcrho.vars,
    # rsnr.vars,
    # rsnr.snr_class,
    ropdp.vars,
    # rnme.vars,
    # rozdr.vars,
    # rmlyr.mlyr_limits,
    # rattc.vars,
    # rd_qcatc.vars,
    # var2plot='snr [dB]',
    # var2plot='cmap [0-1]',
    # var2plot='rhoHV [-]',
    # var2plot='KDP* [deg/km]',
    # var2plot='alpha [-]',
    # var2plot='Rainfall [mm/h]',
    # var2plot='AH [dB/km]',
    var2plot='PhiDP [deg]',
    # var2plot='PIA [dB]',
    # var2plot='ZDR [dB]',
    # var2plot='ZH+ [dBZ]',
    # ylims={'ZH [dBZ]': (0, 50)},
    # radial_xlims=(10, 62.5),
    vars_bounds={
        # 'alpha [-]': (0.01, .18, 18),
        # 'PhiDP [deg]': (80, 220, 13),
        # 'Rainfall [mm/h]': [0.1, 128, 11],
        # 'ZH+ [dBZ]': [5, 50, 10],
        'PIA [dB]': [0, 19, 20],
        'snr [dB]': [-50, 150, 41*2-1],
        'KDP [deg/km]': (-0.5, 1.5, 17),
        # 'PhiDP [deg]': (-10, 85, 20)
        },
    # radial_ylims={'PhiDP [deg]': (-5, 270),}
    # ucmap='tpylc_div_yw_gy_bu',
    # proj='polar',
    # cbticks=rmlyr.regionID,
    # mlyr=rmlyr
    # ppi_xlims=[-40, 40], ppi_ylims=[-40, 40]
    # mlyr=rmlyr,
    )
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
