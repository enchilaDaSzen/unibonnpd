#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:55:54 2022

@author: dsanchez
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import towerpy as tp
import wradlib as wrl
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
from radar import twpext as tpx
from radar.rparams_dwdxpol import RPARAMS

# =============================================================================
# %% Define date-time and working directories
# =============================================================================
START_TIME = dt.datetime(2023, 8, 6, 0, 0)
# START_TIME = dt.datetime(2024, 3, 11, 12, 0)
# START_TIME = dt.datetime(2025, 7, 29, 0, 0)
STOP_TIME = START_TIME+dt.timedelta(hours=24)
# START_TIME = dt.datetime(2023, 3, 8, 1, 0)

PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/enchiladasz/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'
PROFSDATA = EWDIR + 'pd_rdres/20210714/'
MFS_DIR = (LWDIR + 'codes/github/unibonnpd/qc/xpol_mfs/')
CLM_DIR = (LWDIR + 'codes/github/unibonnpd/qc/xpol_clm/')

# =============================================================================
# %% Define radar features and list files
# =============================================================================
RSITE = 'Aaxpol'
RPARAMS = {rs['site_name']: rs for rs in RPARAMS if rs['site_name'] == RSITE}

SCAN_ELEVS = {'el_450': 'sweep_8', 'el_250': 'sweep_7', 'el_124': 'sweep_6',
              'el_105': 'sweep_5', 'el_086': 'sweep_0', 'el_067': 'sweep_1',
              'el_048': 'sweep_2', 'el_029': 'sweep_3', 'el_010': 'sweep_4'}
SCAN_ELEV = SCAN_ELEVS.get('el_010')

LPFILES = tpx.get_listfilesxpol(RSITE, START_TIME, STOP_TIME, parent_dir=None)

# =============================================================================
# %% Import data from wradlib to towerpy
# =============================================================================
N = 70
N = 85
PLOT_METHODS = False

rdata = tpx.Rad_scan(LPFILES[N], f'{RSITE}')
rdata.ppi_xpol(scan_elev=SCAN_ELEV)

tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                   rdata.vars)

# =============================================================================
# %% Radar reflectivity $(Z_H)$ offset correction
# =============================================================================
# rdata.zh_offset = RPARAMS[RSITE]['zhO']
rdata.zh_offset = 0.
rdata.vars['ZH [dBZ]'] += rdata.zh_offset
print(f'{rdata.site_name}_ZH_O [{rdata.zh_offset :.2f} dBZ]')

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rdata.vars)

# =============================================================================
# %% Correlation coefficient $(\rho_{HV})$ noise-correction
# =============================================================================
rcrho = tpx.rhoHV_Noise_Bias(rdata)
# rhohv_theo, noise_lvl = (0.90, 1.1), None
rcrho.iterate_radcst(rdata.georef, rdata.params, rdata.vars, noise_lvl=None,
                     rhohv_theo=RPARAMS[rcrho.site_name]['rhvtc'],
                     data2correct=rdata.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rdata.vars, rcrho.vars,
        var2plot1='rhoHV [-]', var2plot2='rhoHV [-]',
        ucmap_diff='tpylsc_div_dbu_rd', diff_lims=[-0.5, 0.5, .1])

# =============================================================================
# %% Noise suppression
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
if rdata.params['radar constant [dB]'] <= 0:
    min_snr = -rcrho.rhohv_corrs['Noise level [dB]']
else:
    min_snr = rcrho.rhohv_corrs['Noise level [dB]']
print(f"minSNR = {min_snr:.2f} dB")
rsnr.signalnoiseratio(rdata.georef, rdata.params, rcrho.vars,
                      min_snr=-rcrho.rhohv_corrs['Noise level [dB]'],
                      data2correct=rcrho.vars, plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rsnr.vars)

# =============================================================================
# %% Differential phase $(\Phi_{DP})$ quality control and processing
# =============================================================================
ropdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)

# %%% $\Phi_{DP}(0)$ detection and correction
presetphidp = RPARAMS[ropdp.site_name]['phidp_prst'].get(
    START_TIME.strftime("%Y%m%d"))
ropdp.offsetdetection_ppi(rsnr.vars, preset=presetphidp)
print(f'Phi_DP(0) = {ropdp.phidp_offset:.2f}')
ropdp.offset_correction(
    rsnr.vars['PhiDP [deg]'], phidp_offset=ropdp.phidp_offset,
    data2correct=rsnr.vars)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='PhiDP [deg]')

# %%% $\Phi_{DP}$ unfolding
ropdp.vars['PhiDP [deg]'] = np.ascontiguousarray(
    wrl.dp.unfold_phi(ropdp.vars['PhiDP [deg]'],
                      ropdp.vars['rhoHV [-]'],
                      width=3, copy=True).astype(np.float64))

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='PhiDP [deg]')

# =============================================================================
# %% Non-meteorological echoes identification and removal
# =============================================================================
rnme = tp.eclass.nme.NME_ID(rdata)
clmap = None
bcode = RPARAMS[rnme.site_name]['bclass']

# %%% Despeckle and removal of linear signatures
rnme.lsinterference_filter(rdata.georef, rdata.params, ropdp.vars,
                           data2correct=ropdp.vars, plot_method=PLOT_METHODS)

# %%% Clutter ID and removal
rnme.clutter_id(rdata.georef, rdata.params, rnme.vars, binary_class=bcode,
                min_snr=rsnr.min_snr, clmap=clmap, data2correct=rnme.vars,
                plot_method=PLOT_METHODS)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='rhoHV [-]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars,
                                    var2plot='rhoHV [-]')

# ============================================================================
# %% Melting layer allocation
# ============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 1.9
rmlyr.ml_thickness = 0.85
rmlyr.ml_bottom = rmlyr.ml_top-rmlyr.ml_thickness
rmlyr.ml_ppidelimitation(rdata.georef, rdata.params, rnme.vars,
                         plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rnme.vars,
                                       mlyr=rmlyr)

# =============================================================================
# %% Differential reflectivity $(Z_{DR})$ offset correction
# =============================================================================
rozdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rozdr.offset_correction(rnme.vars['ZDR [dB]'], zdr_offset=-1.35,
                        data2correct=rnme.vars)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(rdata.georef, rdata.params, rdata.vars,
                                        rozdr.vars, var2plot1='ZDR [dB]',
                                        var2plot2='ZDR [dB]',
                                        diff_lims=[-1, 1, 0.1])

# =============================================================================
# %% Radar reflectivity $(Z_H)$ attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)

# %%% PHI processing for attenuation correction
rattc.attc_phidp_prepro(rdata.georef, rdata.params, rozdr.vars, rhohv_min=0.85,
                        phidp0_correction=False)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
                                    var2plot='PhiDP [deg]')

# %%% ZH attenuation correction
att_alphax = [0.15, 0.28, 0.22]  # Light rain PARK
att_alphax = [0.28, 0.4, 0.34]  # Moderate to heavy rain PARK
# att_alphax = [0.15, 0.35, 0.4, 0.34]  # PARK

rattc.zh_correction(rdata.georef, rdata.params, rattc.vars,
                    rnme.nme_classif['classif [EC]'], mlyr=rmlyr,
                    attc_method='ABRI', pdp_pxavr_azm=3, pdp_dmin=1,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    coeff_a=[9.781e-5, 1.749e-4, 1.367e-4],  # Park
                    coeff_b=[0.757, 0.804, 0.78],  # Park
                    # coeff_a=[5.50e-5, 1.62e-4, 9.745e-05],  # Diederich
                    # coeff_b=[0.74, 0.86, 0.8],  # Diederich
                    coeff_alpha=att_alphax,
                    plot_method=PLOT_METHODS)

# =============================================================================
# %% Computation of $Z_H(A_H)$ (for PBB, wet radome and miscalibration)
# =============================================================================
temp = 15
rzhah = tp.attc.r_att_refl.Attn_Refl_Relation(rdata)
rzhah.ah_zh(rattc.vars, rband='X', zh_lower_lim=20, zh_upper_lim=55, temp=temp,
            copy_ofr=True, plot_method=PLOT_METHODS)
rattc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']

mov_avrgf_len = (1, 5)
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
    tp.datavis.rad_display.plot_ppidiff(rdata.georef, rdata.params, rattc.vars,
                                        rattc.vars, var2plot1='ZH [dBZ]',
                                        var2plot2='ZH+ [dBZ]')

# =============================================================================
# %% ZDR attenuation correction
# =============================================================================
rb_a = 0.14  # Tropical
rb_a = 0.19  # Continental

rattc.zdr_correction(rdata.georef, rdata.params, rozdr.vars, rattc.vars,
                     rnme.nme_classif['classif [EC]'], mlyr=rmlyr, descr=True,
                     rhv_thld=0.95, mov_avrgf_len=7, minbins=10, p2avrf=5,
                     coeff_beta=[0.02, 0.1, 0.06], beta_alpha_ratio=rb_a,
                     attc_method='BRI', zh_zdr_model='exp',
                     rparams={'coeff_a': RPARAMS[rattc.site_name]['zdrzh_a'],
                              'coeff_b': RPARAMS[rattc.site_name]['zdrzh_b']},
                     plot_method=PLOT_METHODS)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(rdata.georef, rdata.params, rozdr.vars,
                                        rattc.vars, var2plot1='ZDR [dB]',
                                        var2plot2='ZDR [dB]',
                                        diff_lims=[-5, 5, .5])

# =============================================================================
# %% Specific differential phase $(K_{DP})$ calculation
# =============================================================================
# KDP Vulpiani
zh_kdp = 'ZH+ [dBZ]'
rkdpv = {}
kdp_vulp = kdpvpi(rattc.vars['PhiDP [deg]'], winlen=3,
                  dr=rdata.params['gateres [m]']/1000, copy=True)
rkdpv['PhiDP [deg]'] = kdp_vulp[0]
rkdpv['KDP [deg/km]'] = kdp_vulp[1]

# Remove NME
rattc.vars['KDP* [deg/km]'] = np.where(rnme.nme_classif['classif [EC]'] != 0,
                                       np.nan, rkdpv['KDP [deg/km]'])
# rattc.vars['KDP* [deg/km]'] = np.where(rnme.ls_dsp_class['classif [EC]'] != 0,
#                                        np.nan, rkdpv['KDP [deg/km]'])
rattc.vars['PhiDPv [deg]'] = np.where(rnme.nme_classif['classif [EC]'] != 0,
                                      np.nan, rkdpv['PhiDP [deg]'])

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
# rattc.vars['PhiDPv [deg]'] = rkdpv['PhiDPv [deg]']

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rattc.vars, rattc.vars,
        var2plot1='KDP* [deg/km]', var2plot2='KDP+ [deg/km]',
        diff_lims=[-5, 5, .5], vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

# =============================================================================
# %% Rainfall estimation
# =============================================================================
zh_r = 'ZH+ [dBZ]'  # ZH(AH)
kdp_r = 'KDP+ [deg/km]'  # Vulpiani+AH
zdr_r = 'ZDR [dB]'  # ZDR_ATTC
zh_kdpo = 'ZH+ [dBZ]'  # ZH(AH)
zh_zho = 'ZH+ [dBZ]'  # ZH(AH)
kdp_kdpo = 'KDP [deg/km]'  # KDP(AH)

z_thld = 40
thr_zwsnw = 0
thr_zhail = 55

rz_a, rz_b = 72, 2.14  # Diederich2015
# rz_a, rz_b = (1/0.098)**(1/0.57), 1/0.47  # Chen2021
# rz_a, rz_b = (1/0.057)**(1/0.57), 1/0.57  # Chen2023
rz_ahail, rz_bhail = (1/0.035)**(1/0.52), 1/0.52  # Chen2023
rkdp_a, rkdp_b = 16.9, 0.801  # Diederich2015
# rkdp_a, rkdp_b = 22.9, 0.76  # Chen2023
# rkdp_a, rkdp_b = 15.6, 0.83  # Chen2021
rkdp_zdr_a, rkdp_zdr_b, rkdp_zdr_c = 28.6, 0.95, -1.37  # Bringi

# %%% Traditional estimators
rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe.adp_to_r(rattc.vars['ADP [dB/km]'], rband='X', temp=temp, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], rband='X', temp=temp, mlyr=rmlyr,
             beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(rattc.vars[kdp_r], a=rkdp_a, b=rkdp_b, mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.z_to_r(rattc.vars[zh_r], a=rz_a, b=rz_b, mlyr=rmlyr,
            beam_height=rdata.georef['beam_height [km]'])

# %%% Hybrid estimators
rqpe.kdp_zdr_to_r(rattc.vars[kdp_r], rattc.vars[zdr_r],
                  a=rkdp_zdr_a, b=rkdp_zdr_b, c=rkdp_zdr_c,
                  mlyr=rmlyr, beam_height=rdata.georef['beam_height [km]'])

rqpe.z_ah_to_r(rattc.vars[zh_r], rattc.vars['AH [dB/km]'], rband='X',
               rz_a=rz_a, rz_b=rz_b, temp=temp, z_thld=z_thld, mlyr=rmlyr,
               beam_height=rdata.georef['beam_height [km]'])
rqpe.z_kdp_to_r(rattc.vars[zh_r], rattc.vars[kdp_r], rz_a=rz_a, rz_b=rz_b,
                rkdp_a=rkdp_a, rkdp_b=rkdp_b, mlyr=rmlyr, z_thld=z_thld,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r(rattc.vars[zh_r], rattc.vars[zdr_r], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])

# %%% Adaptive estimators
rzh_fit = tpx.rzh_opt(rattc.vars[zh_zho], rqpe.r_ah, rattc.vars['AH [dB/km]'],
                      pia=rattc.vars['PIA [dB]'], maxpia=20, rzfit_b=1.6,
                      rz_stv=[[rz_a, rz_b]], plot_method=PLOT_METHODS)
rqpe_opt = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_opt.z_to_r(rattc.vars[zh_r], a=rzh_fit[0], b=rzh_fit[1], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.r_zopt = rqpe_opt.r_z
rkdp_fit = tpx.rkdp_opt(rattc.vars[kdp_kdpo], rattc.vars[zh_kdpo], mlyr=rmlyr,
                        plot_method=True)
rqpe_opt.kdp_to_r(rattc.vars[kdp_r], a=rkdp_fit[0], b=rkdp_fit[1], mlyr=rmlyr,
                  beam_height=rdata.georef['beam_height [km]'])
rqpe.r_kdpopt = rqpe_opt.r_kdp

# =============================================================================
# %%% QPE above the melting layer bottom
# =============================================================================
qpe_amlb = False

if qpe_amlb:
    f_rz_ml = 0.6
    f_rz_sp = 2.8
else:
    f_rz_ml = 0.
    f_rz_sp = 0.
# max_rkm = 151

# %%%% RZ relation is modified by applying a factor to data within the ML.
rqpe_ml = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_ml.z_to_r(rattc.vars[zh_r], a=rz_a, b=rz_b)
rqpe_ml.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 2)
    & (rattc.vars[zh_r] > thr_zwsnw),
    rqpe_ml.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)

# %%%% RZ relation is modified by applying a factor to data above the ML.
rqpe_sp = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_sp.z_to_r(rattc.vars[zh_r], a=rz_a, b=rz_b)
rqpe_sp.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 3.),
    rqpe_sp.r_z['Rainfall [mm/h]']*f_rz_sp, rqpe_ml.r_z['Rainfall [mm/h]'])

# Inset R(ZH)[amlb] into the other rainfall products
[setattr(rqpe, rp, {(k1): (np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
    getattr(rqpe, rp)['Rainfall [mm/h]'],
    rqpe_sp.r_z['Rainfall [mm/h]']) if 'Rainfall' in k1 else v1)
    for k1, v1 in getattr(rqpe, rp).items()})
    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]

# %%%% rz_hail is applied to data below the ML using a threshold in ZH
rqpe_hail = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_hail.z_to_r(rattc.vars[zh_r], a=rz_ahail, b=rz_bhail)
rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rattc.vars[zh_r] >= thr_zhail),
    rqpe_hail.r_z['Rainfall [mm/h]'], 0)

# Inset R(ZH)[hail] into the other rainfall products
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
                                    xlims=[4.3, 9.2], ylims=[52.75, 48.75],
                                    # NRW
                                    # xlims=[4.5, 16.5], ylims=[55.5, 46.5]
                                    # DEU
                                    )

# =============================================================================
# %% Data visualisation
# =============================================================================
# Create a Towerpy radar object to manage radar variables, georeferenced grids
# and radar parameters for efficient data analysis.
rd_qcatc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rd_qcatc.georef = rdata.georef
rd_qcatc.params = rdata.params
# rd_qcatc.vars = dict(rattc.vars)
# rd_qcatc.alpha_ah = np.nanmean([np.nanmean(i)
#                                 for i in rattc.vars['alpha [-]']])
rd_qcatc.vars = {}
# rd_qcat.vars['ZH [dBZ]'] = rattc.vars['ZH [dBZ]']
rd_qcatc.vars['ZH+ [dBZ]'] = rattc.vars['ZH+ [dBZ]']
rd_qcatc.vars['ZDR [dB]'] = rattc.vars['ZDR [dB]']
rd_qcatc.vars['PhiDP [deg]'] = rattc.vars['PhiDP [deg]']
rd_qcatc.vars['rhoHV [-]'] = rozdr.vars['rhoHV [-]']
rd_qcatc.vars['AH [dB/km]'] = rattc.vars['AH [dB/km]']
rd_qcatc.vars['KDP+ [deg/km]'] = rattc.vars['KDP+ [deg/km]']
# rd_qcatc.vars['alpha [-]'] = rattc.vars['alpha [-]']
# rd_qcatc.vars['Beam_height [km]'] = rdata.georef['beam_height [km]']

# %%%  Plot processed radar variables
tp.datavis.rad_display.plot_setppi(rd_qcatc.georef, rd_qcatc.params,
                                   rd_qcatc.vars, mlyr=rmlyr)

# %%%  Plot cone coverage
tp.datavis.rad_display.plot_cone_coverage(
    rd_qcatc.georef, rd_qcatc.params, rd_qcatc.vars, limh=12, zlims=[0, 12],
    # cbticks=rmlyr.regionID, ucmap='tpylc_div_yw_gy_bu'
    var2plot='KDP+ [deg/km]')

# %%% Interactive plot
tp.datavis.rad_interactive.ppi_base(rd_qcatc.georef, rd_qcatc.params,
                                    rd_qcatc.vars)
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
