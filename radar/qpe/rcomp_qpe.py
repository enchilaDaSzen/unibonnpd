#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:09:09 2024

@author: dsanchez
"""
import datetime as dt
from time import perf_counter
import sys
import os
import pickle
import copy
import numpy as np
import towerpy as tp
from tqdm import tqdm
from towerpy.datavis import rad_display
import wradlib as wrl
import cartopy.crs as ccrs
# LWDIR = '/home/dsanchez/sciebo_dsr/'
LWDIR = '/home/enchiladaszen/Documents/sciebo/'
sys.path.append(LWDIR + 'codes/github/unibonnpd/')
from radar.rparams_dwdxpol import RPARAMS
from radar import twpext as tpx

# =============================================================================
# Define working directory, time and list files
# =============================================================================
data4calib = 'qvps'

# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24h []
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24h [NO JXP]
START_TIME = dt.datetime(2018, 12, 2, 0, 0)  # 24 h [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24h [NO JXP]
# # # START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 8, 0)  # 16h [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24h [NO BXP]

EVNTD_HRS = (16 if START_TIME == dt.datetime(2019, 7, 20, 8, 0) else 24)

STOP_TIME = START_TIME + dt.timedelta(hours=EVNTD_HRS)

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
LWDIR = '/home/enchiladaszen/Documents/sciebo/'
EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

# =============================================================================

# RSITES = ['Boxpol', 'Juxpol']
# COMPNAME = 'XPol'
# RPARAMS = [rs for rs in RPARAMS if rs['site_name'] in RSITES]
# RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
#            + "/rcomp_qpe_xpol/5min/")
# comp_x = [171369.04139240552, 513676.24503467174]  # XPOL
# comp_y = [5471843.801871114, 5794840.492845044]  # XPOL
# nbins = 1000

# =============================================================================

# RSITES = ['Essen', 'Flechtdorf', 'Neuheilenbach', 'Offenthal']
# COMPNAME = 'DWD4'
# RPARAMS = [rs for rs in RPARAMS if rs['site_name'] in RSITES]
# RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
#            + "/rcomp_qpe_dwd/5min/")
# comp_x = [174861.61516327888, 635954.0753363081]  # DWD4
# comp_y = [5387250.290467849, 5846678.789985427]  # DWD4
# # nbins = 600

# =============================================================================

# RSITES = ['Boxpol', 'Juxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach',
#           'Offenthal']
# COMPNAME = 'DWD4+XPol'
# RPARAMS = [rs for rs in RPARAMS if rs['site_name'] in RSITES]
# RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
#            + "/rcomp_qpe_dwdxpol/5min/")
# comp_x = [171369.04139240552, 635954.0753363081]  # DWD4XPOL
# comp_y = [5387250.290467849, 5846678.789985427]  # DWD4XPOL
# # nbins = 1000

# =============================================================================

RSITES = ['Boxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach', 'Offenthal']
COMPNAME = 'DWD4+BXP'
RPARAMS = [rs for rs in RPARAMS if rs['site_name'] in RSITES]
RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
           + "/rcomp_qpe_dwdbxp/5min/")
comp_x = [174861.61516327888, 635954.0753363081]  # DWDBXP
comp_y = [5387250.290467849, 5846678.789985427]  # DWDBXP
# nbins = 750

# =============================================================================

# RSITES = ['Juxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach', 'Offenthal']
# COMPNAME = 'DWD4+JXP'
# RPARAMS = [rs for rs in RPARAMS if rs['site_name'] in RSITES]
# RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
#            + "/rcomp_qpe_dwdjxp/5min/")
# comp_x = [171369.04139240552, 635954.0753363081]  # DWDJXP
# comp_y = [5387250.290467849, 5846678.789985427]  # DWDJXP
# # nbins = 1000

# =============================================================================

nbins = 500

# =============================================================================
# Read-in QVPs data
# =============================================================================
PROFSDATA = LWDIR + f"pd_rdres/qvps_d4calib/{START_TIME.strftime('%Y%m%d')}/"

RCAL_FILES = {rs['site_name']:
              [PROFSDATA+n for n in sorted(os.listdir(PROFSDATA))
              if data4calib in n and rs['site_name'] in n]
              for rs in RPARAMS}

profs_data = {}
for k1, rs in RCAL_FILES.items():
    with open(rs[0], 'rb') as breader:
        profs_data[k1] = pickle.load(breader)

mlt_avg = {k1: np.nanmean([i.ml_top for i in profs_data[k1]['mlyr']])
           for k1, v1 in profs_data.items()}
mlk_avg = {k1: np.nanmean([i.ml_thickness for i in profs_data[k1]['mlyr']])
           for k1, v1 in profs_data.items()}

# =============================================================================
# Set plotting parameters
# =============================================================================
xlims, ylims = [4.3, 11.], [48.5, 52.8]  # DWDXPOL NRW
PLOT_METHODS = False
fig_size = (13, 7)

# %%
# =============================================================================
# List QC radar data
# =============================================================================
RATTCQC_DIR = {rs['site_name']:
               EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}/"
               + f"rsite_qc/{rs['site_name']}/"
               for rs in RPARAMS}

RATTCQC = {k1: [i for i in sorted(os.listdir(rs)) if i.endswith('_rdqc.tpy')]
           for k1, rs in RATTCQC_DIR.items()}

# Check that date-time of the scans are within a given time window.
DELTA_TIME = dt.timedelta(minutes=5)

rs_ts = {k1: np.array([dt.datetime.strptime(v2[:v2.find('_')],
                                            '%Y%m%d%H%M%S%f')
                       for v2 in v1]) for k1, v1 in RATTCQC.items()}
rs_fts = {k1: tpx.fill_timeseries(rs_ts[k1],
                                  range(len(rs_ts[k1])),
                                  stspdt=(START_TIME, STOP_TIME),
                                  toldt=dt.timedelta(minutes=2))[1]
          for k1, v1 in RATTCQC.items()}
# RATTCQC = {k1: [RES_DIR[k1]+i for i in rs]
#                 for k1, rs in rs_fts.items()}
RATTCQC = {k1: [RATTCQC_DIR[k1]+RATTCQC[k1][i] if ~np.isnan(i)
                else np.nan for i in rs] for k1, rs in rs_fts.items()}
RATTCQC = [list(j) for j in zip(*RATTCQC.values())]

# %%
# =============================================================================
# Set QPE parameters
# =============================================================================
rprods_dp = ['r_adp', 'r_ah', 'r_kdp', 'r_z']
rprods_hbr = ['r_ah_kdp', 'r_kdp_zdr', 'r_z_ah', 'r_z_kdp', 'r_z_zdr']
rprods_opt = ['r_kdpopt', 'r_zopt']
rprods_hyop = ['r_ah_kdpopt', 'r_zopt_ah', 'r_zopt_kdp', 'r_zopt_kdpopt']
rprods_nmd = ['r_aho_kdpo', 'r_kdpo', 'r_zo', 'r_zo_ah', 'r_zo_kdp',
              'r_zo_zdr']

# rprods = sorted(rprods_dp + rprods_hbr + rprods_opt + rprods_hyop
#                 + ['r_zo', 'r_kdpo', 'r_aho_kdpo', 'r_kdpo2'])
rprods = sorted(rprods_dp + rprods_hbr + rprods_opt + rprods_hyop
                + ['r_zo', 'r_kdpo', 'r_aho_kdpo'])

z_thld = 40
temp = 15
qpe_amlb = False
if qpe_amlb:
    RES_DIR = RES_DIR.replace('5min/', '5min_amlb/')

# =============================================================================
# # define target grid for composition
# =============================================================================
cgrid_x = np.linspace(comp_x[0] - nbins, comp_x[1] + nbins, int(nbins*2))
cgrid_y = np.linspace(comp_y[0] - nbins, comp_y[1] + nbins, int(nbins*2))
grid_coords = wrl.util.gridaspoints(cgrid_y, cgrid_x)

# %%
# =============================================================================
# Compute QPE and build composite
# =============================================================================
tic = perf_counter()

rqpe_dt = []
r_georef = []
r_params = []

# ratcf = RATTCQC[210]  # Use to evaluate single case
# ratcf = RATTCQC[0]  # Use to evaluate single case
for cntf, ratcf in enumerate(
        tqdm(RATTCQC, desc=f"{START_TIME.strftime('%Y%m%d')}"
             + f" -- Computing RQPE [{COMPNAME}]")):
    resattc = {}
    for rf in ratcf:
        if rf is not np.nan:
            with open(rf, 'rb') as fpkl:
                resattc1 = pickle.load(fpkl)
                resattc[resattc1.site_name] = resattc1
    rmlyr = {k1: tp.ml.mlyr.MeltingLayer(robj) for k1, robj in resattc.items()}
    for k1, robj in rmlyr.items():
        robj.ml_top = resattc[k1].ml_top
        robj.ml_bottom = resattc[k1].ml_bottom
        robj.ml_thickness = resattc[k1].ml_thickness
    [robj.ml_ppidelimitation(resattc[k1].georef, resattc[k1].params,
                             resattc[k1].vars, plot_method=PLOT_METHODS)
     for k1, robj in rmlyr.items()]
    if PLOT_METHODS:
        v2p = 'ZH+ [dBZ]'
        [tp.datavis.rad_display.plot_ppi(
            robj.georef, robj.params, resattc[k1].vars, var2plot=v2p,
            cpy_feats={'status': True}, data_proj=ccrs.PlateCarree(),
            proj_suffix='wgs84', fig_size=fig_size, xlims=xlims, ylims=ylims)
         for k1, robj in resattc.items()]
    rbands = {k1: next(item['rband'] for item in RPARAMS
                       if item['site_name'] == robj.site_name)
              for k1, robj in resattc.items()}
    adpr = 'ADP [dB/km]'
    ahr = 'AH [dB/km]'
    kdpr = 'KDP+ [deg/km]'  # Vulpiani+AH
    zdrr = 'ZDR [dB]'
    zh_kdpo = {k1: 'ZH+ [dBZ]' if rb == 'X' else 'ZH+ [dBZ]'
               for k1, rb in rbands.items()}
    zh_zho = {k1: 'ZH+ [dBZ]' if rb == 'X' else 'ZH+ [dBZ]'
              for k1, rb in rbands.items()}
    zh_r = {k1: 'ZH+ [dBZ]' if rb == 'X' else 'ZH+ [dBZ]'
            for k1, rb in rbands.items()}
    kdp_kdpo = {k1: 'KDP+ [deg/km]' if rb == 'X' else 'KDP+ [deg/km]'
                for k1, rb in rbands.items()}
    r_coeffs = {robj.site_name: {} for k1, robj in resattc.items()}
    for k1, robj in resattc.items():
        if rbands[robj.site_name] == 'C':
            if START_TIME == dt.datetime(2021, 7, 14, 0, 0):
                r_coeffs[k1]['rz_a'] = (1/0.026)**(1/0.69)
                r_coeffs[k1]['rz_b'] = 1/0.69  # Chen2023
                r_coeffs[k1]['rkdp_a'] = 30.6
                r_coeffs[k1]['rkdp_b'] = 0.71  # Chen2023
                r_coeffs[k1]['rah_a'] = 427
                r_coeffs[k1]['rah_b'] = 0.94  # Chen2023
            else:
                r_coeffs[k1]['rz_a'] = (1/0.052)**(1/0.57)
                r_coeffs[k1]['rz_b'] = 1/0.57  # Chen2021
                r_coeffs[k1]['rkdp_a'] = 20.7
                r_coeffs[k1]['rkdp_b'] = 0.72  # Chen2021
                r_coeffs[k1]['rah_a'] = 307
                r_coeffs[k1]['rah_b'] = 0.92  # Chen2021
        elif rbands[k1] == 'X':
            if START_TIME == dt.datetime(2021, 7, 14, 0, 0):
                r_coeffs[k1]['rz_a'] = (1/0.057)**(1/0.57)
                r_coeffs[k1]['rz_b'] = 1/0.57  # Chen2023
                r_coeffs[k1]['rkdp_a'] = 22.9
                r_coeffs[k1]['rkdp_b'] = 0.76  # Chen2023
                r_coeffs[k1]['rah_a'] = 67
                r_coeffs[k1]['rah_b'] = 0.78  # Chen2023
            else:
                r_coeffs[k1]['rz_a'] = (1/0.098)**(1/0.47)
                r_coeffs[k1]['rz_b'] = 1/0.47  # Chen2021
                r_coeffs[k1]['rkdp_a'] = 15.6
                r_coeffs[k1]['rkdp_b'] = 0.83  # Chen2021
                r_coeffs[k1]['rah_a'] = 38
                r_coeffs[k1]['rah_b'] = 0.69  # Chen2021

    # =============================================================================
    # Rainfall estimators
    # =============================================================================
    rqpe = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in resattc.items()}
    if 'r_adp' in rprods:
        [robj.adp_to_r(resattc[k1].vars[adpr], mlyr=rmlyr[k1], temp=temp,
                       rband=rbands[robj.site_name],
                       beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_ah' in rprods:
        [robj.ah_to_r(resattc[k1].vars[ahr], mlyr=rmlyr[k1], temp=temp,
                      rband=rbands[robj.site_name],
                      beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_kdp' in rprods:
        [robj.kdp_to_r(resattc[k1].vars[kdpr], mlyr=rmlyr[k1],
                       a=next(item['rkdp_a'] for item in RPARAMS
                              if item['site_name'] == robj.site_name),
                       b=next(item['rkdp_b'] for item in RPARAMS
                              if item['site_name'] == robj.site_name),
                       beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_z' in rprods:
        [robj.z_to_r(resattc[k1].vars[zh_r[robj.site_name]], mlyr=rmlyr[k1],
                     a=next(item['rz_a'] for item in RPARAMS
                            if item['site_name'] == robj.site_name),
                     b=next(item['rz_b'] for item in RPARAMS
                            if item['site_name'] == robj.site_name),
                     beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    # =============================================================================
    # Hybrid estimators
    # =============================================================================
    if 'r_kdp_zdr' in rprods:
        [robj.kdp_zdr_to_r(resattc[k1].vars[kdpr], resattc[k1].vars[zdrr],
                           mlyr=rmlyr[k1],
                           a=next(item['rkdpzdr_a'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                           b=next(item['rkdpzdr_b'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                           c=next(item['rkdpzdr_c'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                           beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_z_ah' in rprods:
        [robj.z_ah_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                        resattc[k1].vars[ahr], mlyr=rmlyr[k1],
                        rz_a=next(item['rz_a'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                        rz_b=next(item['rz_b'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                        # rz_a=(1/0.026)**(1/0.69), rz_b=1/0.69,
                        rband=rbands[robj.site_name], temp=temp, z_thld=z_thld,
                        beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_z_kdp' in rprods:
        [robj.z_kdp_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                         resattc[k1].vars[kdpr], z_thld=z_thld, mlyr=rmlyr[k1],
                         rz_a=next(item['rz_a'] for item in RPARAMS
                                   if item['site_name'] == robj.site_name),
                         rz_b=next(item['rz_b'] for item in RPARAMS
                                   if item['site_name'] == robj.site_name),
                         rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                                     if item['site_name'] == robj.site_name),
                         rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                                     if item['site_name'] == robj.site_name),
                         beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_z_zdr' in rprods:
        [robj.z_zdr_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                         resattc[k1].vars[zdrr], mlyr=rmlyr[k1],
                         a=next(item['rzhzdr_a'] for item in RPARAMS
                                if item['site_name'] == robj.site_name),
                         b=next(item['rzhzdr_b'] for item in RPARAMS
                                if item['site_name'] == robj.site_name),
                         c=next(item['rzhzdr_c'] for item in RPARAMS
                                if item['site_name'] == robj.site_name),
                         beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    if 'r_ah_kdp' in rprods:
        [robj.ah_kdp_to_r(
            resattc[k1].vars[zh_r[robj.site_name]], resattc[k1].vars[ahr],
            resattc[k1].vars[kdpr], mlyr=rmlyr[k1], temp=temp, z_thld=z_thld,
            rband=rbands[robj.site_name],
            # rah_a=r_coeffs[robj.site_name]['rah_a'],
            # rah_b=r_coeffs[robj.site_name]['rah_b'],
            # rkdp_a=r_coeffs[robj.site_name]['rkdp_a'],
            # rkdp_b=r_coeffs[robj.site_name]['rkdp_b'],
            rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                        if item['site_name'] == robj.site_name),
            rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                        if item['site_name'] == robj.site_name),
            beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe.items()]
    # =============================================================================
    # Adaptive estimators
    # =============================================================================
    rqpe_opt = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
                for k1, robj in resattc.items()}
    rqpe_opt2 = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
                 for k1, robj in resattc.items()}
    if 'r_kdpopt' in rprods:
        rkdp_fit = {k1: tpx.rkdp_opt(
            resattc[k1].vars[kdp_kdpo[robj.site_name]],
            resattc[k1].vars[zh_kdpo[robj.site_name]], mlyr=rmlyr[k1],
            rband=rbands[robj.site_name], kdpmed=0.5,
            zh_thr=((40, 50) if rbands[robj.site_name] == 'X'
                    else (44.5, 45.5)),
            rkdp_stv=(next(item['rkdp_a'] for item in RPARAMS
                           if item['site_name'] == robj.site_name),
                      next(item['rkdp_b'] for item in RPARAMS
                           if item['site_name'] == robj.site_name)),
            plot_method=PLOT_METHODS)
                    for k1, robj in rqpe_opt.items()}
        [robj.kdp_to_r(resattc[k1].vars[kdpr], mlyr=rmlyr[k1],
                       a=rkdp_fit[k1][0], b=rkdp_fit[k1][1],
                       beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_opt.items()]
        for k1, robjz in rqpe.items():
            robjz.r_kdpopt = rqpe_opt[k1].r_kdp
    if 'r_zopt' in rprods:
        rzh_fit = {k1: tpx.rzh_opt(
            resattc[k1].vars[zh_zho[robj.site_name]], rqpe[k1].r_ah,
            resattc[k1].vars['AH [dB/km]'], pia=resattc[k1].vars['PIA [dB]'],
            mlyr=rmlyr[k1],
            maxpia=(50 if rbands[robj.site_name] == 'X' else 50),
            # minpia=(0.1 if rbands[robj.site_name] == 'X' else 0.1),
            rzfit_b=(2.14 if rbands[robj.site_name] == 'X' else 1.6),
            rz_stv=[next(item['rz_a'] for item in RPARAMS
                         if item['site_name'] == robj.site_name),
                    next(item['rz_b'] for item in RPARAMS
                         if item['site_name'] == robj.site_name)],
            plot_method=PLOT_METHODS) for k1, robj in rqpe_opt.items()}
        [robj.z_to_r(resattc[k1].vars[zh_r[robj.site_name]], mlyr=rmlyr[k1],
                     a=rzh_fit[k1][0], b=rzh_fit[k1][1],
                     beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_opt.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zopt = rqpe_opt[k1].r_z
    if 'r_zopt_ah' in rprods and 'r_zopt' in rprods:
        [robj.z_ah_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                        resattc[k1].vars[ahr], rz_a=rzh_fit[k1][0],
                        rz_b=rzh_fit[k1][1], rband=rbands[robj.site_name],
                        temp=temp, z_thld=z_thld, mlyr=rmlyr[k1],
                        beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_opt.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zopt_ah = rqpe_opt[k1].r_z_ah
    if 'r_zopt_kdp' in rprods and 'r_zopt' in rprods:
        [robj.z_kdp_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                         resattc[k1].vars[kdpr], z_thld=z_thld, mlyr=rmlyr[k1],
                         rz_a=rzh_fit[k1][0], rz_b=rzh_fit[k1][1],
                         rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                                     if item['site_name'] == robj.site_name),
                         rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                                     if item['site_name'] == robj.site_name),
                         beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_opt.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zopt_kdp = rqpe_opt[k1].r_z_kdp
    if ('r_zopt_kdpopt' in rprods and 'r_zopt' in rprods
            and 'r_kdpopt' in rprods):
        [robj.z_kdp_to_r(resattc[cnt].vars[zh_r[robj.site_name]],
                         resattc[cnt].vars[kdpr], z_thld=z_thld,
                         mlyr=rmlyr[cnt],
                         rz_a=rzh_fit[cnt][0], rz_b=rzh_fit[cnt][1],
                         rkdp_a=rkdp_fit[cnt][0], rkdp_b=rkdp_fit[cnt][1],
                         beam_height=resattc[cnt].georef['beam_height [km]'])
         for cnt, robj in rqpe_opt2.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zopt_kdpopt = rqpe_opt2[k1].r_z_kdp
    if 'r_ah_kdpopt' in rprods:
        [robj.ah_kdp_to_r(
            resattc[k1].vars[zh_r[robj.site_name]], resattc[k1].vars[ahr],
            resattc[k1].vars[kdpr], mlyr=rmlyr[k1], temp=temp, z_thld=z_thld,
            rband=rbands[robj.site_name],
            rkdp_a=rkdp_fit[k1][0], rkdp_b=rkdp_fit[k1][1],
            beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_opt.items()]
        for k1, robjz in rqpe.items():
            robjz.r_ah_kdpopt = rqpe_opt[k1].r_ah_kdp
    # =============================================================================
    # Estimators using non-fully corrected variables
    # =============================================================================
    rqpe_nfc = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
                for k1, robj in resattc.items()}
    rqpe_nfc2 = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
                 for k1, robj in resattc.items()}
    kdpr2 = 'KDP* [deg/km]'  # Vulpiani
    zhr2 = 'ZH [dBZ]'  # ZHattc
    if 'r_kdpo' in rprods:
        [robj.kdp_to_r(resattc[k1].vars[kdpr2], mlyr=rmlyr[k1],
                       a=r_coeffs[robj.site_name]['rkdp_a'],
                       b=r_coeffs[robj.site_name]['rkdp_b'],
                       beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_nfc.items()]
        for k1, robjk in rqpe.items():
            robjk.r_kdpo = rqpe_nfc[k1].r_kdp
    if 'r_kdpo2' in rprods:
        if START_TIME == dt.datetime(2021, 7, 14, 0, 0):
            [robj.kdp_to_r(resattc[k1].vars[kdpr2], mlyr=rmlyr[k1],
                           a=next(item['rkdp_a'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                           b=next(item['rkdp_b'] for item in RPARAMS
                                  if item['site_name'] == robj.site_name),
                           beam_height=resattc[k1].georef['beam_height [km]'])
             for k1, robj in rqpe_nfc2.items()]
            for k1, robjk in rqpe.items():
                robjk.r_kdpo2 = rqpe_nfc2[k1].r_kdp
    if 'r_zo' in rprods:
        [robj.z_to_r(resattc[k1].vars[zhr2], mlyr=rmlyr[k1],
                     a=r_coeffs[robj.site_name]['rz_a'],
                     b=r_coeffs[robj.site_name]['rz_b'],
                     beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_nfc.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zo = rqpe_nfc[k1].r_z
    if 'r_zo_ah' in rprods:
        [robj.z_ah_to_r(resattc[k1].vars[zhr2], resattc[k1].vars[ahr],
                        rband=rbands[robj.site_name], temp=temp, z_thld=z_thld,
                        mlyr=rmlyr[k1],
                        rz_a=r_coeffs[robj.site_name]['rz_a'],
                        rz_b=r_coeffs[robj.site_name]['rz_b'],
                        rah_a=r_coeffs[robj.site_name]['rah_a'],
                        rah_b=r_coeffs[robj.site_name]['rah_b'],
                        beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_nfc.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zo_ah = rqpe_nfc[k1].r_z_ah
    if 'r_zo_kdp' in rprods:
        [robj.z_kdp_to_r(resattc[k1].vars[zhr2], resattc[k1].vars[kdpr2],
                         z_thld=z_thld, mlyr=rmlyr[k1],
                         rz_a=r_coeffs[robj.site_name]['rz_a'],
                         rz_b=r_coeffs[robj.site_name]['rz_b'],
                         rkdp_a=r_coeffs[robj.site_name]['rkdp_a'],
                         rkdp_b=r_coeffs[robj.site_name]['rkdp_b'],
                         beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_nfc.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zo_kdp = rqpe_nfc[k1].r_z_kdp
    if 'r_zo_zdr' in rprods:
        [robj.z_zdr_to_r(resattc[k1].vars[zhr2], resattc[k1].vars[zdrr],
                         mlyr=rmlyr[k1],
                         a=next(item['rzhzdr_a'] for item in RPARAMS
                                if item['site_name'] == robj.site_name),
                         b=next(item['rzhzdr_b'] for item in RPARAMS
                                if item['site_name'] == robj.site_name),
                         c=next(item['rzhzdr_c'] for item in RPARAMS
                                if item['site_name'] == robj.site_name),
                         beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_nfc.items()]
        for k1, robjz in rqpe.items():
            robjz.r_zo_zdr = rqpe_nfc[k1].r_z_zdr
    if 'r_aho_kdpo' in rprods:
        [robj.ah_kdp_to_r(
            resattc[k1].vars[zhr2], resattc[k1].vars[ahr],
            resattc[k1].vars[kdpr2], mlyr=rmlyr[k1], temp=temp, z_thld=z_thld,
            rband=rbands[robj.site_name],
            rah_a=r_coeffs[robj.site_name]['rah_a'],
            rah_b=r_coeffs[robj.site_name]['rah_b'],
            rkdp_a=r_coeffs[robj.site_name]['rkdp_a'],
            rkdp_b=r_coeffs[robj.site_name]['rkdp_b'],
            beam_height=resattc[k1].georef['beam_height [km]'])
         for k1, robj in rqpe_nfc.items()]
        for k1, robjz in rqpe.items():
            robjz.r_aho_kdpo = rqpe_nfc[k1].r_ah_kdp
    # =============================================================================
    # QPE within and above the MLYR
    # =============================================================================
    if qpe_amlb:
        thr_zwsnw = 0
        thr_zhail = 55
        f_rz_ml = 0.6
        f_rz_aml = 2.8
    else:
        thr_zwsnw = 0
        thr_zhail = 55
        f_rz_ml = 0
        f_rz_aml = 0
    # =============================================================================
    # RZ relation is modified by applying a factor to data within the ML.
    rqpe_ml = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
               for k1, robj in resattc.items()}
    [robj.z_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                 a=next(item['rz_a'] for item in RPARAMS
                        if item['site_name'] == robj.site_name),
                 b=next(item['rz_b'] for item in RPARAMS
                        if item['site_name'] == robj.site_name))
     for k1, robj in rqpe_ml.items()]
    for k1, robj in rqpe_ml.items():
        robj.r_z['Rainfall [mm/h]'] = np.where(
            (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 2)
            & (resattc[k1].vars[zh_r[robj.site_name]] > thr_zwsnw),
            robj.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)
    # =============================================================================
    # RZ relation is modified by applying a factor to data above the ML.
    rqpe_aml = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
                for k1, robj in resattc.items()}
    [robj.z_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                 a=next(item['rz_a'] for item in RPARAMS
                        if item['site_name'] == robj.site_name),
                 b=next(item['rz_b'] for item in RPARAMS
                        if item['site_name'] == robj.site_name))
     for k1, robj in rqpe_aml.items()]
    for k1, robj in rqpe_aml.items():
        robj.r_z['Rainfall [mm/h]'] = np.where(
            (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 3.),
            robj.r_z['Rainfall [mm/h]']*f_rz_aml,
            rqpe_ml[k1].r_z['Rainfall [mm/h]'])
    # Correct all other variables
    for k1, robj in rqpe.items():
        [setattr(robj, rp, {(k2): (np.where(
            (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1),
            getattr(robj, rp)['Rainfall [mm/h]'],
            rqpe_aml[k1].r_z['Rainfall [mm/h]']) if 'Rainfall' in k2 else v1)
            for k2, v1 in getattr(robj, rp).items()})
            for rp in robj.__dict__.keys() if rp.startswith('r_')]
    # =============================================================================
    # rz_hail is applied to data below the ML with Z > 55 dBZ
    rqpe_hail = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
                 for k1, robj in resattc.items()}
    [robj.z_to_r(resattc[k1].vars[zh_r[robj.site_name]],
                 a=next(item['rz_haila'] for item in RPARAMS
                        if item['site_name'] == robj.site_name),
                 b=next(item['rz_hailb'] for item in RPARAMS
                        if item['site_name'] == robj.site_name))
     for k1, robj in rqpe_hail.items()]
    for k1, robj in rqpe_hail.items():
        robj.r_z['Rainfall [mm/h]'] = np.where(
            (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1)
            & (resattc[k1].vars[zh_r[robj.site_name]] >= thr_zhail),
            robj.r_z['Rainfall [mm/h]'], np.nan)
    # # Correct all other variables
    for k1, robj in rqpe.items():
        [setattr(
            robj, rp, {(k2): (np.where(
                (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1)
                & (resattc[k1].vars[zh_r[robj.site_name]] >= thr_zhail),
                rqpe_hail[k1].r_z['Rainfall [mm/h]'],
                getattr(robj, rp)['Rainfall [mm/h]']) if 'Rainfall' in k2
                else v1) for k2, v1 in getattr(robj, rp).items()})
            for rp in robj.__dict__.keys() if rp.startswith('r_')]
    # Replace 0 with nan for better compositing
    for k1, robj in rqpe.items():
        [setattr(
            robj, rp, {(k1): (
                np.where(getattr(robj, rp)['Rainfall [mm/h]'] == 0, np.nan,
                         getattr(robj, rp)['Rainfall [mm/h]'])
                if 'Rainfall' in k1 else v1)
                for k1, v1 in getattr(robj, rp).items()})
            for rp in robj.__dict__.keys() if rp.startswith('r_')]
    rsite_rcoeffs = {k1: {k2: rv2 if not isinstance(rv2, dict)
                          else {k3: v3 for k3, v3 in rv2.items()
                                if k3 != 'Rainfall [mm/h]'}
                          for k2, rv2 in vars(v1).items()}
                     for k1, v1 in rqpe.items()}
    # for robj in resattc:
    #     robj.georef['beam_height [km]'] += robj.params['altitude [m]']/1000

    if resattc:
        rcomp_params = copy.copy(resattc[list(resattc)[0]].params)
        dt_mean1 = [robj.scandatetime for k1, robj in rqpe.items()]
        dt_mean = min(dt_mean1)+(max(dt_mean1)-min(dt_mean1))/2
        rqpe_dt.append(dt_mean)
        with open(RES_DIR + 'rcoeffs/' + dt_mean.strftime('%Y%m%d%H%M%S_')
                  + 'rsite_qpe_rcoeffs.tpy', 'wb') as f:
            pickle.dump(rsite_rcoeffs, f, pickle.HIGHEST_PROTOCOL)
        # nbins = max([robj.params['ngates'] for k1, robj in resattc.items()])
        # Use of UTM Zone 29 coordinates of range-bin centroids for composition
        rads_coord = {k1: np.array([robj.georef['grid_utmx'].flatten(),
                      robj.georef['grid_utmy'].flatten()]).T
                      for k1, robj in resattc.items()}
        # derive quality information - in this case, the pulse volume
        pulse_volumes = {k1: np.tile(wrl.qual.pulse_volume(
            robj.georef['range [m]'], robj.params['gateres [m]'],
            robj.params['beamwidth [deg]']), robj.params['nrays'])
            for k1, robj in resattc.items()}
        # interpolate polar radar-data and quality data to the grid
        rd_quality_gridded = [wrl.comp.togrid(
            robj, grid_coords, resattc[k1].georef['range [m]'].max()
            + resattc[k1].georef['range [m]'][0], robj.mean(axis=0),
            pulse_volumes[k1], wrl.ipol.Nearest)
            for k1, robj in rads_coord.items()]
        # Define the radar rainfall products to be composed.
        rfields = {k1: {k: v for k, v in robj.__dict__.items()
                        if isinstance(v, dict)} for k1, robj in rqpe.items()}
        rd_gridded = {k1: [wrl.comp.togrid(
            robj, grid_coords, resattc[k2].georef['range [m]'].max()
            + resattc[k2].georef['range [m]'][0], robj.mean(axis=0),
            rfields[k2][k1]['Rainfall [mm/h]'][:, 0:].ravel(),
            wrl.ipol.Nearest)
            for k2, robj in rads_coord.items()] for k1 in rprods}
        # rfieldk = set([k for i in rfields for k in i.keys()])
        # rd_gridded = {k1: [wrl.comp.togrid(
        #     robj, grid_coords, resattc[cnt].georef['range [m]'].max()
        #     + resattc[cnt].georef['range [m]'][0], robj.mean(axis=0),
        #     rfields[cnt][k1]['Rainfall [mm/h]'][:, 0:].ravel(),
        #     wrl.ipol.Nearest)  # wrl.ipol.Idw
        #     for cnt, robj in enumerate(rads_coord)]
        #     for k1 in rfieldk if k1 in rprods}

        # Creates dict including params
        # rcomp_params = copy.copy(resattc[list(resattc)[0]].params)
        # dt_mean1 = [robj.scandatetime for k1, robj in rqpe.items()]
        # dt_mean = min(dt_mean1)+(max(dt_mean1)-min(dt_mean1))/2
        # rqpe_dt.append(dt_mean)

        # rcomps['datetime'] = dt_mean
        # rcomps['elev_ang [deg]'] = {robj.site_name: robj.elev_angle
        #                             for cnt, robj in enumerate(rqpe)}
        r_params.append({'datetime': dt_mean,
                         'elevs': {robj.site_name: robj.elev_angle
                                   for k1, robj in rqpe.items()},
                         'elev_ang [deg]': 'Composite'})
        # compose the r-qpe based on the quality information calculated above
        rcomps = {k1: wrl.comp.compose_weighted(
            v1, [1. / (i + 0.001) for i in rd_quality_gridded])
            for k1, v1 in rd_gridded.items()}
        rcomps = {k1: np.ma.masked_invalid(v1)
                  for k1, v1 in rcomps.items()}
        rcomps = {k1: v1.reshape((len(cgrid_x), len(cgrid_x)))
                  for k1, v1 in rcomps.items()}
        for rp in rprods:
            rprod2save = {rp: rcomps[rp]}
            rprod2save['datetime'] = dt_mean
            rprod2save['elev_ang [deg]'] = {robj.site_name: robj.elev_angle
                                            for k1, robj in rqpe.items()}
            with open(RES_DIR+dt_mean.strftime('%Y%m%d%H%M%S_')
                      + f'rcomp_rqpe_{rp}.tpy', 'wb') as f:
                pickle.dump(rprod2save, f, pickle.HIGHEST_PROTOCOL)
    # Create composite grid from the fist mosaick
    if cntf == 0:
        rcomp_georef = {'grid_utmx':
                        grid_coords[:, 0].reshape(
                            rcomps[list(rcomps.keys())[0]].shape),
                        'grid_utmy':
                            grid_coords[:, 1].reshape(
                                rcomps[list(rcomps.keys())[0]].shape)}
        epsg_to_osr = 32632
        wgs84 = wrl.georef.get_default_projection()
        utm = wrl.georef.epsg_to_osr(epsg_to_osr)
        rcomp_georef['grid_wgs84x'], rcomp_georef['grid_wgs84y'] = (
            wrl.georef.reproject(rcomp_georef['grid_utmx'],
                                 rcomp_georef['grid_utmy'],
                                 src_crs=utm, trg_crs=wgs84))
        with open(RES_DIR+dt_mean.strftime('%Y%m%d')+'_mgrid.tpy',
                  'wb') as f:
            pickle.dump(rcomp_georef, f, pickle.HIGHEST_PROTOCOL)
    if PLOT_METHODS:
        rad_display.plot_ppi(
            rcomp_georef, rcomp_params,
            {ki + 'Rainfall [mm/h]': vi
             for ki, vi in rcomps.items() if ki.startswith('r_')},
            var2plot='r_zoptRainfall [mm/h]', cpy_feats={'status': True},
            # proj_suffix='utm', data_proj=ccrs.UTM(zone=32),
            proj_suffix='wgs84', data_proj=ccrs.PlateCarree(),
            xlims=xlims, ylims=ylims, fig_size=fig_size,
            fig_title=(
                'Radar Composite: '
                + f"{rcomp_params['datetime']:%Y-%m-%d %H:%M}"))

    # print(dt_mean.strftime('%Y%m%d%H%M') + ' --- DONE')
with open(RES_DIR+dt_mean.strftime('%Y%m%d_') + 'params.tpy', 'wb') as f:
    pickle.dump(r_params, f, pickle.HIGHEST_PROTOCOL)
toc1 = perf_counter()
