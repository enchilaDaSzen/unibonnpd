#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:39:20 2021

@author: ds17589
"""

import datetime as dt
import numpy as np
import wradlib as wrl
import towerpy as tp
import cartopy.crs as ccrs
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
from radar import twpext as tpx


# =============================================================================
# Define working directory and list files
# =============================================================================
RSITE = 'chenies'
MODER = 'zdr'  # zdr or ldr
MODEP = 'lp'  # sp or lp
ELEV = 0
START_TIME = dt.datetime(2020, 10, 3, 7, 25)
# START_TIME = dt.datetime(2020, 10, 3, 3, 31)
# START_TIME = dt.datetime(2022, 11, 17, 0, 5)

EWDIR = ('/media/enchiladaszen/Samsung1TB/safe/radar_datasets/'
         f'ukmo-nimrod/data/single-site/{START_TIME.year}/{RSITE}/'
         + f'{MODEP}el{ELEV}/')

FRADNAME = (EWDIR + f'metoffice-c-band-rain-radar_{RSITE}_'
            f'{START_TIME.strftime("%Y%m%d%H%M")}_raw-dual-polar'
            + f'-aug{MODER}-{MODEP}-el{ELEV}.dat')

PLOT_METHODS = False

# =============================================================================
# Reads polar radar data
# =============================================================================
rdata = tp.io.ukmo.Rad_scan(FRADNAME, RSITE)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.ppi_ukmogeoref()

tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# %%
# =============================================================================
# rhoHV noise correction
# =============================================================================
rcrho = tpx.rhoHV_Noise_Bias(rdata)
rhohv_theo = (0.92, 1.1) if MODEP == 'lp' else (0.95, 1.1)
rcrho.iterate_radcst(rdata.georef, rdata.params, rdata.vars,
                     rhohv_theo=rhohv_theo, data2correct=rdata.vars,
                     # noise_lvl=(39, 42, .1),
                     plot_method=PLOT_METHODS)

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
if rdata.params['radar constant [dB]'] < 0:
    min_snr = -rcrho.rhohv_corrs['Noise level [dB]'] + 0
else:
    min_snr = rcrho.rhohv_corrs['Noise level [dB]'] - 0
if MODEP == 'sp':
    min_snr *= 2
print(f"minSNR = {min_snr:.2f} dB")
rsnr.signalnoiseratio(rdata.georef, rdata.params, rcrho.vars, min_snr=min_snr,
                      data2correct=rcrho.vars, plot_method=True)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                       rsnr.vars)

# %%
# =============================================================================
# PhiDP offset correction and unfolding
# =============================================================================
ropdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)
ropdp.offsetdetection_ppi(rsnr.vars, preset=None)
print(f'Phi_DP(0) = {np.median(ropdp.phidp_offset):.2f}')

ropdp.offset_correction(rsnr.vars['PhiDP [deg]'],
                        phidp_offset=ropdp.phidp_offset,
                        data2correct=rsnr.vars)

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
# Classification of non-meteorological echoes
# =============================================================================
if MODEP == 'sp':
    bclass = 191
    clmapf = None
elif MODEP == 'lp':
    bclass = 159
    if RSITE == 'chenies':
        bclass = 223
        clmapf = np.loadtxt('/home/enchiladaszen/Documents/sciebo/codes/'
                            + 'github/towerpy/towerpy/eclass/ukmo_cmaps/'
                            + 'chenies/chenies_cluttermap_el0.dat')
if PLOT_METHODS:
    tp.datavis.rad_display.plot_mfs(
        '/home/enchiladaszen/Documents/sciebo/'
        + 'codes/github/towerpy/towerpy/eclass/mfs_cband/')

rnme = tp.eclass.nme.NME_ID(ropdp)

rnme.lsinterference_filter(rdata.georef, rdata.params, ropdp.vars,
                           rhv_min=0.3, data2correct=ropdp.vars,
                           plot_method=PLOT_METHODS)

rnme.clutter_id(rdata.georef, rdata.params, rnme.vars, binary_class=bclass,
                min_snr=rsnr.min_snr, clmap=clmapf,
                data2correct=rnme.vars, plot_method=PLOT_METHODS)

# %%
# =============================================================================
# Melting layer allocation
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 2.3
rmlyr.ml_thickness = 0.85
rmlyr.ml_bottom = rmlyr.ml_top - rmlyr.ml_thickness

rmlyr.ml_ppidelimitation(rdata.georef, rdata.params, rsnr.vars,
                         plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars,
                                    mlyr=rmlyr, var2plot='rhoHV [-]')

# %%
# =============================================================================
# ZDR offset correction
# =============================================================================
rozdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rozdr.offset_correction(rnme.vars['ZDR [dB]'], zdr_offset=-0.18,
                        data2correct=rnme.vars)
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                    rnme.vars, var2plot='ZDR [dB]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='ZDR [dB]')
# %%
# =============================================================================
# ZH Attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)

rattc.attc_phidp_prepro(rdata.georef, rdata.params, rozdr.vars, rhohv_min=0.85,
                        phidp0_correction=False)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
                                    var2plot='PhiDP [deg]')

rattc.zh_correction(rdata.georef, rdata.params, rattc.vars,
                    rnme.nme_classif['classif [EC]'], mlyr=rmlyr, pdp_dmin=1,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    attc_method='ABRI', pdp_pxavr_azm=3,
                    # coeff_alpha=[0.05, 0.1, 0.08],
                    coeff_alpha=[0.08, 0.18, 0.11],
                    # coeff_alpha=[0.05, 0.18, 0.11],
                    plot_method=PLOT_METHODS)

# %%
# =============================================================================
# PBBc and ZHAH
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
rb_a = 0.39  # Continental
# rb_a = 0.14  # Tropical

rattc.zdr_correction(rdata.georef, rdata.params, rozdr.vars, rattc.vars,
                     rnme.nme_classif['classif [EC]'], mlyr=rmlyr, descr=True,
                     coeff_beta=[0.002, 0.07, 0.04], beta_alpha_ratio=rb_a,
                     rhv_thld=0.98, mov_avrgf_len=9, minbins=5, p2avrf=3,
                     attc_method='BRI', plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(rdata.georef, rdata.params, rozdr.vars,
                                        rattc.vars, var2plot1='ZDR [dB]',
                                        var2plot2='ZDR [dB]',
                                        diff_lims=[-1, 1, .1])

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
rattc.vars['KDP* [deg/km]'] = np.where(rnme.nme_classif['classif [EC]'] != 0,
                                       np.nan, rkdpv['KDP [deg/km]'])
# rattc.vars['KDP* [deg/km]'] = np.where(rnme.ls_dsp_class['classif [EC]'] != 0,
                                       # np.nan, rkdpv['KDP [deg/km]'])

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
zhr_kdp = 'ZH [dBZ]'  # ZH(ATTC)
zhropt = 'ZH [dBZ]'  # ZH(ATTC)
zhr = 'ZH+ [dBZ]'  # ZH(AH)
zdrr = 'ZDR [dB]'
kdpr = 'KDP+ [deg/km]'
z_thld = 40

# rz_a, rz_b = (1/0.052)**(1/0.57), 1/0.57  # Chen2021
# rz_a, rz_b = (1/0.026)**(1/0.69), 1/0.69  # Chen2023
# rkdp_a, rkdp_b = 20.7, 0.72  # Chen2021
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
rqpe.kdp_to_r(rattc.vars[kdpr], mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_zdr_to_r(rattc.vars[kdpr], rattc.vars[zdrr], mlyr=rmlyr,
                  beam_height=rdata.georef['beam_height [km]'])
rqpe.z_to_r(rattc.vars[zhr], mlyr=rmlyr,
            beam_height=rdata.georef['beam_height [km]'])
rqpe.z_ah_to_r(rattc.vars[zhr], rattc.vars['AH [dB/km]'],
               temp=temp, z_thld=z_thld, mlyr=rmlyr,
               beam_height=rdata.georef['beam_height [km]'])
rqpe.z_kdp_to_r(rattc.vars[zhr], rattc.vars[kdpr], mlyr=rmlyr, z_thld=z_thld,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r(rattc.vars[zhr], rattc.vars[zdrr], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
# ZH(R) Adaptive
rzh_fit = tpx.rzh_opt(
    rattc.vars[zhropt], rqpe.r_ah, rattc.vars['AH [dB/km]'], mlyr=rmlyr,
    pia=rattc.vars['PIA [dB]'], maxpia=10, plot_method=PLOT_METHODS)
rqpe_opt = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_opt.z_to_r(rattc.vars[zhr], a=rzh_fit[0], b=rzh_fit[1], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.r_z_opt = rqpe_opt.r_z
# RKDP Adaptive
rkdp_fit = tpx.rkdp_opt(rattc.vars[kdpr], rattc.vars[zhr_kdp], mlyr=rmlyr,
                        rband='C', plot_method=PLOT_METHODS)
rqpe_opt.kdp_to_r(
    rattc.vars[kdpr], a=rkdp_fit[0], b=rkdp_fit[1],
    mlyr=rmlyr, beam_height=rdata.georef['beam_height [km]'])
rqpe.r_kdp_opt = rqpe_opt.r_kdp

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
rqpe_ml.z_to_r(rattc.vars[zhr])
rqpe_ml.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 2)
    & (rattc.vars[zhr] > thr_zwsnw),
    rqpe_ml.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)
# =============================================================================
# Additional factor for the RZ relation is applied to data above the ML
# =============================================================================
rqpe_sp = tp.qpe.qpe_algs.RadarQPE(rdata)
rqpe_sp.z_to_r(rattc.vars[zhr])
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
rqpe_hail.z_to_r(rattc.vars[zhr], a=rz_ahail, b=rz_bhail)
rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rattc.vars[zhr] >= thr_zhail),
    rqpe_hail.r_z['Rainfall [mm/h]'], 0)
# Correct all other variables
[setattr(rqpe, rp, {(k1): (np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rattc.vars[zhr] >= thr_zhail), rqpe_hail.r_z['Rainfall [mm/h]'],
    getattr(rqpe, rp)['Rainfall [mm/h]']) if 'Rainfall' in k1 else v1)
    for k1, v1 in getattr(rqpe, rp).items()})
    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]

if PLOT_METHODS:
    rests = [i for i in sorted(dir(rqpe)) if i.startswith('r_')]
    for i in rests:
        tp.datavis.rad_display.plot_ppi(
            rdata.georef, rdata.params, getattr(rqpe, i),
            fig_title=f"{rdata.params['elev_ang [deg]']:{2}.{3}} deg. --"
            + f" {rdata.params['datetime']:%Y-%m-%d %H:%M:%S} -- {i}")
        tp.datavis.rad_display.plot_ppi(
            rdata.georef, rdata.params, rqpe.r_z_ah,
            cpy_feats={'status': True}, data_proj=ccrs.OSGB(approx=False),
            fig_size=(11, 6))

# %%
# =============================================================================
# OUTPUT
# =============================================================================
rd_qcatc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rd_qcatc.georef = rdata.georef
rd_qcatc.params = rdata.params
rd_qcatc.vars = dict(rattc.vars)

rd_qcatc.vars['KDP [deg/km]'] = rd_qcatc.vars['KDP+ [deg/km]']
rd_qcatc.vars['ZH [dBZ]'] = rd_qcatc.vars['ZH+ [dBZ]']
del rd_qcatc.vars['alpha [-]']
del rd_qcatc.vars['beta [-]']
del rd_qcatc.vars['ZH+ [dBZ]']
del rd_qcatc.vars['ZH* [dBZ]']
# del rd_qcatc.vars['PIA [dB]']
# del rd_qcatc.vars['PhiDP [deg]']
del rd_qcatc.vars['PhiDP* [deg]']
# del rd_qcatc.vars['ADP [dB/km]']
# del rd_qcatc.vars['AH [dB/km]']
del rd_qcatc.vars['KDP+ [deg/km]']
del rd_qcatc.vars['KDP* [deg/km]']
# rd_qcatc.vars['AH [dB/km]'] =
# rd_qcatc.vars['KDP* [deg/km]'] = rkdpv['KDP [deg/km]']

rd_qcatc.vars['rhoHV [-]'] = rozdr.vars['rhoHV [-]']

rd_qcatc.vars['Rainfall [mm/h]'] = rqpe.r_z_opt['Rainfall [mm/h]']

if PLOT_METHODS:
    tp.datavis.rad_display.plot_setppi(rd_qcatc.georef, rd_qcatc.params,
                                       rd_qcatc.vars, mlyr=rmlyr)


# %%
# =============================================================================
# Plots
# =============================================================================
if PLOT_METHODS:
    # Plot cone coverage
    tp.datavis.rad_display.plot_cone_coverage(rdata.georef, rdata.params,
                                              rmlyr.mlyr_limits, limh=12,
                                              zlims=[0, 12],
                                              cbticks=rmlyr.regionID,
                                              ucmap='tpylc_div_yw_gy_bu')

# %%
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params,
                                    rdata.vars,  # proj='polar',
                                    var2plot='PhiDP [deg]', mlyr=rmlyr)
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()

# %%

# =============================================================================
# Read RG data
# =============================================================================
# LWDIR = '/home/dsanchez/sciebo_dsr/'
# EWDIR = '/run/media/dsanchez/enchiladasz/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'


# RG_WDIR = (LWDIR + 'towerpy/midas_rg/')
# UKRG_MDFN = (RG_WDIR + 'midas-open_uk-hourly-rain-obs_dv-202308_station-metadata.csv')
# # RG_NCDATA = (LWDIR + 'pd_rdres/dwd_rg/'
# #              + 'nrw_20210713_20210715_1h_1hac/')

# =============================================================================
# Init raingauge object
# =============================================================================
# rg_data = tpx.RainGauge(RG_WDIR, nwk_opr='UKMO')

# =============================================================================
# Read metadata of all DWD rain gauges (location, records, etc)
# =============================================================================
# rg_data.get_dwdstn_mdata(UKRG_MDFN, plot_methods=PLOT_METHODS)

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
# rg_data.get_stns_box(rg_data.dwd_stn_mdata, bbox_xlims=(6, 11.),
#                      bbox_ylims=(48.5, 52.8), plot_methods=True,
#                      #surface=qpe_georef,
#                      isrfbins=100, dmax2srfbin=1)

# =============================================================================
# Download DWD rg data
# =============================================================================
# for hour in range(72):
#     start_time = dt.datetime(2021, 7, 13, 0, 0, 0)
#     # print(hour)
#     start_time = start_time + dt.timedelta(hours=hour)
#     print(start_time)
#     stop_time = start_time + dt.timedelta(hours=1)
#     print(stop_time)
#     # start_time = start_time + datetime.timedelta(hours=hour+1)
#     # print(start_time)
#     for station_id in rg_data.stn_near_rad['stations_id']:
#         rg_data.get_dwdstn_nc(station_id, start_time, stop_time,
#                               dir_ncdf=rg_ncdata)

# =============================================================================
# Read DWD rg data
# =============================================================================
# rg_data.get_rgdata(resqpe_accd_params['datetime'], ds_ncdir=RG_NCDATA,
#                    drop_nan=True, ds2read=rg_data.stn_bbox,
#                    ds_tres=dt.timedelta(hours=1),
#                    # dt_bkwd=dt.timedelta(hours=3),
#                    # ds_accum=dt.timedelta(hours=3),
#                    dt_bkwd=dt.timedelta(hours=25),
#                    ds_accum=dt.timedelta(hours=25),
#                    plot_methods=False)
