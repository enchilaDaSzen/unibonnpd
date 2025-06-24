#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:16:22 2022

@author: enchiladaszen
"""
import os
import copy
from time import perf_counter
import datetime as dt
import pickle
import numpy as np
import towerpy as tp
import wradlib as wrl
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
import cartopy.crs as ccrs
from radar import twpext as tpx
from towerpy.datavis import rad_display
from radar.rparams_dwdxpol import RPARAMS


# =============================================================================
# Define working directory, and date-time
# =============================================================================
START_TIME = dt.datetime(2017, 7, 19, 19, 0)  # 12hr [NO JXP]
# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24hr [NO JXP]
START_TIME = dt.datetime(2017, 7, 25, 17, 0)  # 24hr [NO JXP]
START_TIME = dt.datetime(2018, 5, 16, 17, 0)  # 24hr []
START_TIME = dt.datetime(2018, 9, 23, 16, 30)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 14, 30)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2019, 5, 11, 4, 0)  # 24hr [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 14, 30)  # 16 hr [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
START_TIME = dt.datetime(2021, 2, 6, 10, 40)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24 hr []
# START_TIME = dt.datetime(2021, 7, 14, 10, 10)  # 24 hr [NO BXP]
START_TIME = dt.datetime(2021, 7, 14, 17, 40)  # 24 hr [NO BXP]

STOP_TIME = START_TIME+dt.timedelta(minutes=5)

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
PDIR = None
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'
# PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'

MFS_DIR = (LWDIR + 'codes/github/unibonnpd/radar/nme/xpol_mfs/')
CLM_DIR = (LWDIR + 'codes/github/unibonnpd/radar/nme/xpol_clm/')

# =============================================================================
# Define radar sites and list files
# =============================================================================
RSITES = ['Boxpol', 'Juxpol']
RSITES = ['Essen', 'Flechtdorf', 'Neuheilenbach', 'Offenthal']
RSITES = ['Boxpol', 'Juxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach',
          'Offenthal']
RSITES = ['Juxpol', 'Essen']

RPARAMS = {rs['site_name']: rs for rs in RPARAMS if rs['site_name'] in RSITES}

RSITE_FILES = {i['site_name']:
               tpx.get_listfilesxpol(i['site_name'], START_TIME, STOP_TIME,
                                     i['elev'], parent_dir=PDIR)[0]
               if 'xpol' in i['site_name'].lower() else
               tpx.get_listfilesdwd(i['site_name'], START_TIME, STOP_TIME,
                                    i['elev'], parent_dir=PDIR)
               for k, i in RPARAMS.items()}

COMPOSITE = True

# =============================================================================
# Set parameters related to QC
# =============================================================================
max_diffdtmin = 120
qpe_amlb = False

# =============================================================================
# Set plotting parameters
# =============================================================================
PLOT_METHODS = False

# xlims, ylims = [4.3, 9.2], [52.75, 48.75]  # XPOL NRW
# xlims, ylims = [4.3, 16.5], [55.5, 46.5]  # Germany
# xlims, ylims = [4.3, 11.], [48.5, 52.8]  # DWDXPOL COV
# xlims, ylims = [4.15, 11.], [48.55, 52.75]  # DWDXPOL RADCOV
# xlims, ylims = [5.9, 10.5], [49., 52.5]  # PAPER
xlims, ylims = [4.324, 10.953], [48.635, 52.754]  # DWDXPOL RADCOV
fig_size = (13, 7)
xlims, ylims = [5.85, 11.], [48.55, 52.75]  # DWDXPOL DE
fig_size = (10.5, 7)

# %%
# =============================================================================
# Read-in QVPs data
# =============================================================================
data4calib = 'qvps'
appx = ''  # _extmlyr
PROFSDATA = LWDIR + (f"pd_rdres/qvps_d4calib{appx}/"
                     + f"{START_TIME.strftime('%Y%m%d')}/")

RCAL_FILES = {k1: [PROFSDATA+n for n in sorted(os.listdir(PROFSDATA))
              if data4calib in n and k1 in n] for k1, rs in RPARAMS.items()}

profs_data = {}
for k1, rs in RCAL_FILES.items():
    with open(rs[0], 'rb') as breader:
        profs_data[k1] = pickle.load(breader)

mlt_avg = {k1: np.nanmean([i.ml_top for i in profs_data[k1]['mlyr']])
           for k1, v1 in profs_data.items()}
mlk_avg = {k1: np.nanmean([i.ml_thickness for i in profs_data[k1]['mlyr']])
           for k1, v1 in profs_data.items()}

mlyrhv = {k1: [i for i in v1['mlyr'] if ~np.isnan(i.ml_top)]
          for k1, v1 in profs_data.items()}
phidpOv = {k1: [i for i in v1['phidpO'] if ~np.isnan(i.phidp_offset)]
           for k1, v1 in profs_data.items()}
zdrOv = {k1: [i for i in v1['zdrO'] if ~np.isnan(i.zdr_offset)]
         for k1, v1 in profs_data.items()}

mlyrhv = {k1: (v1 if v1 else [profs_data[k1]['mlyr'][0]])
          for k1, v1 in mlyrhv.items()}

# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
tic = perf_counter()
rdata = {k1: tpx.Rad_scan(RSITE_FILES[k1], k1)
         for k1, robj in RPARAMS.items() if RSITE_FILES[k1]}

[robj.ppi_xpol() if robj.site_name.lower() == 'boxpol' else
 robj.ppi_xpol(scan_elev=RPARAMS[k1]['elev'])
 if robj.site_name.lower() == 'juxpol'
 else robj.ppi_dwd() for k1, robj in rdata.items()]

rbands = {k1: RPARAMS[k1]['rband'] for k1, robj in rdata.items()}

if PLOT_METHODS:
    tp.datavis.rad_display.plot_mgrid(
        [i.georef for k, i in rdata.items()],
        [i.params for k, i in rdata.items()],
        [i.vars for k, i in rdata.items()], ncols=3, nrows=2,
        cpy_feats={'status': True}, proj_suffix='utm',
        data_proj=ccrs.UTM(zone=32), xlims=xlims, ylims=ylims)
[tp.datavis.rad_display.plot_setppi(robj.georef, robj.params, robj.vars)
 for k1, robj in rdata.items()]

# %%
# =============================================================================
# Allocate closest dataset from QVPs (for ML, offsets, etc)
# =============================================================================
idx_mlh = {k1: (min(zip(range(
    len([i.scandatetime for i in mlyrhv[k1]])),
        [i.scandatetime for i in mlyrhv[k1]]),
    key=lambda x: (x[1] > robj.scandatetime,
                   abs(x[1] - robj.scandatetime))) if mlyrhv[k1]
    else (0, profs_data[k1]['mlyr'][0].scandatetime))
    for k1, robj in rdata.items()}
idx_phidp0 = {k1: min(zip(range(
    len([i.scandatetime for i in phidpOv[k1]])),
        [i.scandatetime for i in phidpOv[k1]]),
    key=lambda x: (x[1] > robj.scandatetime,
                   abs(x[1] - robj.scandatetime)))
    for k1, robj in rdata.items()}
idx_zdr0 = {k1: min(zip(range(
    len([i.scandatetime for i in zdrOv[k1]])),
        [i.scandatetime for i in zdrOv[k1]]),
    key=lambda x: (x[1] > robj.scandatetime,
                   abs(x[1] - robj.scandatetime)))
    for k1, robj in rdata.items()}
idx_pclass = {k1: min(zip(range(
    len(profs_data[k1]['dtrs'])), profs_data[k1]['dtrs']),
    key=lambda x: (x[1] > robj.scandatetime,
                   abs(x[1] - robj.scandatetime)))
    for k1, robj in rdata.items()}

# nbins = max([robj.params['ngates'] for k1, robj in rdata.items()])
# %%
# =============================================================================
# ZH offset correction
# =============================================================================
for k1, v1 in RPARAMS.items():
    # ZH offset
    if 'xpol' in k1:
        # v1['zhO'] = v1['zh_offset'].get(START_TIME.strftime("%Y%m%d"))
        v1['zhO'] = 0
    else:
        v1['zhO'] = 0

for k1, robj in rdata.items():
    robj.zh_offset = RPARAMS[k1]['zhO']
    robj.vars['ZH [dBZ]'] += robj.zh_offset
    print(f'{robj.site_name}_ZH_O [{robj.zh_offset :.2f} dBZ]')

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars) for k1, robj in rdata.items()]

# %%
# =============================================================================
# rhoHV noise-correction
# =============================================================================
# noise_lvl = None
rcrho = {k1: tpx.rhoHV_Noise_Bias(robj) for k1, robj in rdata.items()}
[robj.iterate_radcst(
    rdata[k1].georef, rdata[k1].params, rdata[k1].vars,  # noise_lvl=noise_lvl,
    rhohv_theo=RPARAMS[k1]['rhvtc'],
    noise_lvl=RPARAMS[k1]['nlvl'].get(START_TIME.strftime("%Y%m%d")),
    data2correct=rdata[k1].vars, plot_method=PLOT_METHODS)
 for k1, robj in rcrho.items()]

for k1, robj in rcrho.items():
    print(f"{robj.site_name}_Noise_Level"
          + f" [{robj.rhohv_corrs['Noise level [dB]'] :.2f} dB]")

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppidiff(
        robj.georef, robj.params, robj.vars, rcrho[k1].vars,
        var2plot1='rhoHV [-]', var2plot2='rhoHV [-]',
        ucmap_diff='tpylsc_div_dbu_rd', diff_lims=[-0.5, 0.5, .1])
     for k1, robj in rdata.items()]

# %%
# =============================================================================
# Noise suppression
# =============================================================================
min_snr = {k1: (-robj.rhohv_corrs['Noise level [dB]']
           if rdata[k1].params['radar constant [dB]'] <= 0
           else robj.rhohv_corrs['Noise level [dB]'])
           for k1, robj in rcrho.items()}
rsnr = {k1: tp.eclass.snr.SNR_Classif(robj) for k1, robj in rcrho.items()}
[robj.signalnoiseratio(rdata[k1].georef, rdata[k1].params, rcrho[k1].vars,
                       min_snr=min_snr[k1], data2correct=rcrho[k1].vars,
                       plot_method=PLOT_METHODS)
 for k1, robj in rsnr.items()]

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars)
     for k1, robj in rsnr.items()]
    [tp.datavis.rad_display.plot_setppi(rdata[k1].georef, rdata[k1].params,
                                        robj.vars)
     for k1, robj in rsnr.items()]

# %%
# =============================================================================
# PhiDP quality control and processing
# =============================================================================
ropdp = {k1: tp.calib.calib_phidp.PhiDP_Calibration(robj)
         for k1, robj in rsnr.items()}
# Modify the PhiDP sign (only for JXP site)
for k1, rprm in RPARAMS.items():
    if rprm['site_name'] == 'Juxpol' and START_TIME.year > 2018:
        rprm['signpdp'] = -1
    else:
        rprm['signpdp'] = 1
for k1, robj in rsnr.items():
    robj.vars['PhiDP [deg]'] *= RPARAMS[k1]['signpdp']
[robj.offsetdetection_ppi(
    rsnr[k1].vars,
    preset=RPARAMS[k1]['phidp_prst'].get(START_TIME.strftime("%Y%m%d")))
    for k1, robj in ropdp.items()]
[robj.offset_correction(rsnr[k1].vars['PhiDP [deg]'],
                        phidp_offset=robj.phidp_offset,
                        data2correct=rsnr[k1].vars)
 for k1, robj in ropdp.items()]

for k1, robj in ropdp.items():
    print(f'{robj.site_name}_PhiDP_O '
          + f'[{robj.phidp_offset:.2f} deg]')
    robj.vars['PhiDP [deg]'] = np.ascontiguousarray(
        wrl.dp.unfold_phi(robj.vars['PhiDP [deg]'], robj.vars['rhoHV [-]'],
                          width=RPARAMS[k1]['wu_pdp'],
                          copy=True)).astype(np.float64)

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars, var2plot='PhiDP [deg]')
     for k1, robj in rsnr.items()]
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars, var2plot='PhiDP [deg]')
     for k1, robj in ropdp.items()]
# %%
# =============================================================================
# Clutter identification and removal
# =============================================================================
rnme = {k1: tp.eclass.nme.NME_ID(robj) for k1, robj in ropdp.items()}
for k1, robj in rnme.items():
    if robj.site_name == 'Boxpol':
        pathmfscl = MFS_DIR
        if robj.scandatetime.year == 2017:
            clfmap = np.loadtxt(
                CLM_DIR + f'{robj.site_name.lower()}{robj.scandatetime.year}b'
                + '_cluttermap_el0.dat')
        else:
            clfmap = np.loadtxt(
                CLM_DIR + f'{robj.site_name.lower()}{robj.scandatetime.year}'
                + '_cluttermap_el0.dat')
        if RPARAMS['Boxpol']['elev'] != 'n_ppi_010deg':
            RPARAMS['Boxpol']['bclass'] -= 64
    elif robj.site_name == 'Juxpol':
        pathmfscl = None
        clfmap = np.loadtxt(CLM_DIR+f'{robj.site_name.lower()}'
                            # + f'{robj.scandatetime.year}'
                            + '2021' + '_cluttermap_el0.dat')
        # clfmap = None
    else:
        pathmfscl = None
        rdata2 = tpx.Rad_scan(RSITE_FILES[robj.site_name], robj.site_name)
        # DWD clutter map is not always available,
        # these lines try to read such data
        try:
            rdata2.ppi_dwd(get_rvar='cmap')
            clfmap = 1 - tp.utils.radutilities.normalisenanvalues(
                rdata2.vars['cmap [0-1]'],
                np.nanmin(rdata2.vars['cmap [0-1]']),
                np.nanmax(rdata2.vars['cmap [0-1]']))
            clfmap = np.nan_to_num(clfmap, nan=1e-5)
            RPARAMS[k1]['bclass'] = 207
            pass
        except Exception:
            clfmap = None
            RPARAMS[k1]['bclass'] = 207 - 64
            print('No CL Map available')
            pass
    robj.lsinterference_filter(rdata[k1].georef, rdata[k1].params,
                               ropdp[k1].vars, data2correct=ropdp[k1].vars,
                               rhv_min=RPARAMS[k1]['rhvmin'],
                               # rhv_min=0.7,
                               plot_method=PLOT_METHODS)
    robj.clutter_id(rdata[k1].georef, rdata[k1].params, robj.vars,
                    path_mfs=pathmfscl, clmap=clfmap,
                    min_snr=rsnr[k1].min_snr, data2correct=robj.vars,
                    binary_class=RPARAMS[k1]['bclass'],
                    plot_method=PLOT_METHODS)

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars)
     for k1, robj in rnme.items()]
    [tp.datavis.rad_display.plot_setppi(rdata[k1].georef, rdata[k1].params,
                                        robj.vars)
     for k1, robj in rnme.items()]
# %%
# ============================================================================
# Melting layer allocation
# ============================================================================
rmlyr = {k1: tp.ml.mlyr.MeltingLayer(robj) for k1, robj in rnme.items()}
for k1, robj in rmlyr.items():
    if robj.site_name in profs_data:
        if ((abs((idx_mlh[k1][1] - robj.scandatetime).total_seconds())/60)
                <= max_diffdtmin):
            robj.ml_top = mlyrhv[k1][idx_mlh[k1][0]].ml_top
            robj.ml_bottom = mlyrhv[k1][idx_mlh[k1][0]].ml_bottom
            robj.ml_thickness = mlyrhv[k1][idx_mlh[k1][0]].ml_thickness
            if (np.isnan(robj.ml_thickness) and ~np.isnan(robj.ml_top)
                    and ~np.isnan(robj.ml_bottom)):
                robj.ml_thickness = robj.ml_top - robj.ml_bottom
            if (np.isnan(mlt_avg[k1]) and np.isnan(mlk_avg[k1])):
                print(f'{robj.site_name} ML_h not in database,'
                      + ' using the preset value')
                robj.ml_top = RPARAMS[k1]['mlt']
                robj.ml_thickness = RPARAMS[k1]['mlk']
                robj.ml_bottom = robj.ml_top-robj.ml_thickness
            else:
                print(f'{robj.site_name} ML_h in database')
        else:
            robj.ml_top = mlt_avg[k1]
            robj.ml_thickness = mlk_avg[k1]
            robj.ml_bottom = robj.ml_top - robj.ml_thickness
            print(f'{robj.site_name} ML_h in database lies too far,'
                  + ' using the day-average value')

[robj.ml_ppidelimitation(rdata[k1].georef, rdata[k1].params, rnme[k1].vars,
                         plot_method=PLOT_METHODS)
 for k1, robj in rmlyr.items()]

if PLOT_METHODS:
    tp.datavis.rad_display.plot_mgrid(
        [i.georef for k, i in rdata.items()],
        [i.params for k, i in rdata.items()],
        [i.mlyr_limits for k, i in rmlyr.items()], ncols=3, nrows=2,
        cpy_feats={'status': True}, proj_suffix='utm',
        data_proj=ccrs.UTM(zone=32), xlims=xlims, ylims=ylims)

# %%
# =============================================================================
# ZDR offset correction
# =============================================================================
rozdr = {k1: tp.calib.calib_zdr.ZDR_Calibration(robj)
         for k1, robj in rmlyr.items()}
for k1, robj in rozdr.items():
    if robj.site_name in profs_data:
        if ((abs((idx_zdr0[k1][1] - robj.scandatetime).total_seconds())/60)
                <= max_diffdtmin):
            robj.zdr_offset = zdrOv[k1][idx_zdr0[k1][0]].zdr_offset
            if robj.zdr_offset == 0 or np.isnan(robj.zdr_offset):
                print(f'{robj.site_name}_ZDR offset void --'
                      + ' using fixed value')
                robj.zdr_offset = RPARAMS[k1]['zdr_offset']
            else:
                print(f'{robj.site_name}_ZDR_O '
                      + f'[{robj.zdr_offset:.2f} dB]')
        else:
            print(f'{robj.site_name}_ZDR offset dt in database'
                  + ' far too long -- using fixed value')
            robj.zdr_offset = RPARAMS[k1]['zdr_offset']
    else:
        print(f'{robj.site_name}_ZDR offset not in database')
[robj.offset_correction(rnme[k1].vars['ZDR [dB]'], zdr_offset=robj.zdr_offset,
                        data2correct=rnme[k1].vars)
 for k1, robj in rozdr.items()]

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppidiff(
        robj.georef, robj.params, rnme[k1].vars, rozdr[k1].vars,
        var2plot1='ZDR [dB]', var2plot2='ZDR [dB]', diff_lims=[-2, 2, .25])
     for k1, robj in rdata.items()]
# %%
# =============================================================================
# ZH attenuation correction
# =============================================================================
rattc = {k1: tp.attc.attc_zhzdr.AttenuationCorrection(robj)
         for k1, robj in rdata.items()}

for k1, robj in rattc.items():
    if robj.site_name in profs_data:
        if ((abs((idx_pclass[k1][1] - robj.scandatetime).total_seconds())/60)
                <= max_diffdtmin):
            robj.pcp_type = profs_data[k1]['pcp_type'][idx_pclass[k1][0]]
        else:
            robj.pcp_type == 1
    if rbands[k1] == 'C' and (robj.pcp_type == 0 or robj.pcp_type == 1
                              or robj.pcp_type == 4):
        robj.att_alphai = [0.05, 0.1, 0.08]  # Light to moderate rain
        robj.att_betaalphar = 0.39  # Continental
        # robj.att_betaalphar = 0.14  # Tropical
        print(f'scan LR-{rbands[k1]}-{robj.att_alphai}')
    elif rbands[k1] == 'C' and (robj.pcp_type == 2 or robj.pcp_type == 5):
        robj.att_alphai = [0.1, 0.18, 0.08]  # Moderate to heavy rain
        robj.att_betaalphar = 0.27  # medval
        print(f'scan MR-{rbands[k1]}-{robj.att_alphai}')
    elif rbands[k1] == 'C' and (robj.pcp_type == 3 or robj.pcp_type == 6):
        robj.att_alphai = [0.1, 0.18, 0.08]  # Moderate to heavy rain
        robj.att_betaalphar = 0.14  # Tropical
        # robj.att_betaalphar = 0.39  # Continental
        # att_alpha = [0.05, 0.18, 0.11]  # Light - heavy rain
        print(f'scan HR-{rbands[k1]}-{robj.att_alphai}')
    if rbands[k1] == 'X' and (robj.pcp_type == 0 or robj.pcp_type == 1
                              or robj.pcp_type == 4):
        robj.att_alphai = [0.15, 0.30, 0.30]  # Light rain PARK
        robj.att_betaalphar = 0.19  # Continental
        # robj.att_betaalphar = 0.14  # Tropical
        print(f'scan LR-{rbands[k1]}-{robj.att_alphai}')
    elif rbands[k1] == 'X' and (robj.pcp_type == 2 or robj.pcp_type == 5):
        robj.att_alphai = [0.30, 0.45, 0.30]  # Moderate to heavy rain PARK
        robj.att_betaalphar = 0.17  # medval
        print(f'scan MR-{rbands[k1]}-{robj.att_alphai}')
    elif rbands[k1] == 'X' and (robj.pcp_type == 3 or robj.pcp_type == 6):
        robj.att_alphai = [0.30, 0.45, 0.30]  # Moderate to heavy rain PARK
        robj.att_betaalphar = 0.14  # Tropical
        # robj.att_betaalphar = 0.19  # Continental
        # att_alpha = [0.15, 0.35, 0.22]  # Light - heavy rain
        print(f'scan HR-{rbands[k1]}-{robj.att_alphai}')

[robj.attc_phidp_prepro(
    rdata[k1].georef, rdata[k1].params, rozdr[k1].vars, rhohv_min=0.85,
    phidp0_correction=(True if (robj.site_name == 'Offenthal')
                       or (robj.site_name == 'Neuheilenbach')
                       or (robj.site_name == 'Flechtdorf')
                       else False))
 for k1, robj in rattc.items()]

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars, var2plot='PhiDP [deg]')
     for k1, robj in rozdr.items()]
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars, var2plot='PhiDP [deg]')
     for k1, robj in rattc.items()]

[robj.zh_correction(rdata[k1].georef, rdata[k1].params, rattc[k1].vars,
                    rnme[k1].nme_classif['classif [EC]'], mlyr=rmlyr[k1],
                    attc_method='ABRI', pdp_dmin=1, pdp_pxavr_azm=3,
                    pdp_pxavr_rng=round(4000/rdata[k1].params['gateres [m]']),
                    coeff_alpha=robj.att_alphai, phidp0=0,
                    # coeff_a=[9.781e-5, 1.749e-4, 1.367e-4],  # Park
                    # coeff_b=[0.757, 0.804, 0.78],  # Park
                    coeff_a=[5.50e-5, 1.62e-4, 9.745e-05],  # Diederich
                    coeff_b=[0.74, 0.86, 0.8],  # Diederich
                    plot_method=PLOT_METHODS)
 if 'xpol' in robj.site_name.lower() else
 robj.zh_correction(rdata[k1].georef, rdata[k1].params, rattc[k1].vars,
                    rnme[k1].nme_classif['classif [EC]'], mlyr=rmlyr[k1],
                    attc_method='ABRI', pdp_dmin=1, pdp_pxavr_azm=3,
                    pdp_pxavr_rng=round(4000/rdata[k1].params['gateres [m]']),
                    coeff_alpha=robj.att_alphai, phidp0=0,
                    # coeff_a=[1.59e-5, 4.27e-5, 2.49e-05],  # Diederich
                    # coeff_b=[0.73, 0.77, 0.755],  # Diederich
                    coeff_a=[1e-5, 4.27e-5, 3e-05],  # MRR+Diederich
                    coeff_b=[0.73, 0.85, 0.78],  # MRR+Diederich
                    plot_method=PLOT_METHODS)
 for k1, robj in rattc.items()]

for k1, robj in rattc.items():
    robj.phidp_offset -= ropdp[k1].phidp_offset

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
                                     robj.vars, var2plot='AH [dB/km]')
     for k1, robj in rattc.items()]

# %%
# =============================================================================
# Partial beam blockage correction
# =============================================================================
temp = 15
rzhah = {k1: tp.attc.r_att_refl.Attn_Refl_Relation(robj)
         for k1, robj in rdata.items()}
zhah_lim = {k1: [20, 55] for k1, robj in rzhah.items()}
[robj.ah_zh(rattc[k1].vars, zh_lower_lim=zhah_lim[k1][0], temp=temp,
            rband=rbands[robj.site_name], zh_upper_lim=zhah_lim[k1][1],
            data2correct=rattc[k1].vars, plot_method=PLOT_METHODS)
 for k1, robj in rzhah.items()]

mov_avrgf_len = (1, 7)

for k1, robj in rattc.items():
    robj.vars['ZH* [dBZ]'] = rzhah[k1].vars['ZH [dBZ]']
    zh_difnan = np.where(rzhah[k1].vars['diff [dBZ]'] == 0,
                         np.nan, rzhah[k1].vars['diff [dBZ]'])
    zhpdiff = np.array([np.nanmedian(i) if ~np.isnan(np.nanmedian(i)) else 0
                        for cnt, i in enumerate(zh_difnan)])
    zhpdiff_pad = np.pad(zhpdiff, mov_avrgf_len[1]//2,
                         mode='wrap')
    zhplus_maf = np.ma.convolve(
        zhpdiff_pad, np.ones(mov_avrgf_len[1])/mov_avrgf_len[1], mode='valid')
    robj.vars['ZH+ [dBZ]'] = np.array(
        [robj.vars['ZH [dBZ]'][cnt] - i if i == 0
         else robj.vars['ZH [dBZ]'][cnt] - zhplus_maf[cnt]
         for cnt, i in enumerate(zhpdiff)])

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppidiff(
        robj.georef, robj.params, rattc[k1].vars, rattc[k1].vars,
        var2plot1='ZH [dBZ]', var2plot2='ZH+ [dBZ]')
     for k1, robj in rdata.items()]

# %%
# =============================================================================
# ZDR attenuation correction
# =============================================================================
zhzdr_a = 0.000249173
zhzdr_b = 2.33327
zdr_attcc = (9, 5, 3)
zdr_attcx = (7, 10, 5)
[robj.zdr_correction(
    rdata[k1].georef, rdata[k1].params, rozdr[k1].vars, rzhah[k1].vars,
    rnme[k1].nme_classif['classif [EC]'], mlyr=rmlyr[k1], attc_method='BRI',
    coeff_beta=RPARAMS[k1]['beta'], beta_alpha_ratio=robj.att_betaalphar,
    rhv_thld=RPARAMS[k1]['rhvatt'],
    mov_avrgf_len=(zdr_attcc[0] if rbands[robj.site_name] == 'C'
                   else zdr_attcx[0]),
    minbins=(zdr_attcc[1] if rbands[robj.site_name] == 'C' else zdr_attcx[1]),
    p2avrf=(zdr_attcc[2] if rbands[robj.site_name] == 'C' else zdr_attcx[2]),
    zh_zdr_model='exp', rparams={'coeff_a': zhzdr_a, 'coeff_b': zhzdr_b},
    plot_method=PLOT_METHODS)
 for k1, robj in rattc.items()]

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppidiff(
        robj.georef, robj.params, rozdr[k1].vars, rattc[k1].vars,
        var2plot1='ZDR [dB]', var2plot2='ZDR [dB]', diff_lims=[-1, 1, .1])
     for k1, robj in rdata.items()]
# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# rkdpv = []
for k1, robj in rattc.items():
    if rbands[robj.site_name] == 'C':
        zh_kdp = 'ZH+ [dBZ]'
    elif rbands[robj.site_name] == 'X':
        zh_kdp = 'ZH+ [dBZ]'
    v = {}
    v['PhiDP [deg]'], v['KDP [deg/km]'] = kdpvpi(
        robj.vars['PhiDP [deg]'], dr=rdata[k1].params['gateres [m]']/1000,
        winlen=RPARAMS[k1]['kdpwl'])
    # Remove NME
    v['KDP* [deg/km]'] = np.where(rnme[k1].nme_classif['classif [EC]'] != 0,
                                  np.nan, v['KDP [deg/km]'])
    # Remove negative KDP values in rain region and within ZH threshold
    robj.vars['KDP* [deg/km]'] = np.where(
        (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1)
        & (v['KDP [deg/km]'] < 0) & (robj.vars[zh_kdp] > 5),  # 0
        0, v['KDP* [deg/km]'])
    v['KDP+ [deg/km]'] = np.where((
        robj.vars[zh_kdp] >= 40) & (robj.vars[zh_kdp] < 55)
        & (rozdr[k1].vars['rhoHV [-]'] >= 0.95)
        & (~np.isnan(robj.vars['KDP [deg/km]']))
        & (robj.vars['KDP [deg/km]'] != 0),
        robj.vars['KDP [deg/km]'], robj.vars['KDP* [deg/km]'])
    robj.vars['KDP+ [deg/km]'] = v['KDP+ [deg/km]']

if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppidiff(
        robj.georef, robj.params, rattc[k1].vars, rattc[k1].vars,
        var2plot1='KDP* [deg/km]', var2plot2='KDP+ [deg/km]',
        diff_lims=[-1, 1, 0.1], vars_bounds={'KDP [deg/km]': (-1.5, 4.5, 17)})
     for k1, robj in rdata.items()]

# %%
# =============================================================================
# Rainfall estimation
# =============================================================================
z_thld = 40
# thr_zwsnw = 0
# thr_zhail = 55

rprods_dp = ['r_adp', 'r_ah', 'r_kdp', 'r_z']
rprods_hbr = ['r_ah_kdp', 'r_kdp_zdr', 'r_z_ah', 'r_z_kdp', 'r_z_zdr']
rprods_opt = ['r_kdpopt', 'r_zopt']
rprods_hyop = ['r_ah_kdpopt', 'r_zopt_ah', 'r_zopt_kdp', 'r_zopt_kdpopt']
rprods_nmd = ['r_aho_kdpo', 'r_kdpo', 'r_zo', 'r_zo_ah', 'r_zo_kdp',
              'r_zo_zdr']

rprods = sorted(rprods_dp[1:] + rprods_hbr + rprods_opt + rprods_hyop
                + ['r_aho_kdpo', 'r_zo', 'r_kdpo'])
# rprods = sorted(['r_z', 'r_kdp'])

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
kdp_kdpo = {k1: 'KDP [deg/km]' if rb == 'X' else 'KDP [deg/km]'
            for k1, rb in rbands.items()}
r_coeffs = {robj.site_name: {} for k1, robj in rattc.items()}
for k1, robj in rattc.items():
    if rbands[robj.site_name] == 'C':
        if START_TIME.date() == dt.date(2021, 7, 14):
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
        if START_TIME.date() == dt.date(2021, 7, 14):
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
# Traditional estimators
# =============================================================================
rqpe = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in rdata.items()}

if 'r_adp' in rprods:
    [robj.adp_to_r(rattc[k1].vars[adpr], mlyr=rmlyr[k1], temp=temp,
                   rband=rbands[robj.site_name],
                   beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_ah' in rprods:
    [robj.ah_to_r(rattc[k1].vars[ahr], mlyr=rmlyr[k1], temp=temp,
                  rband=rbands[robj.site_name],
                  beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_kdp' in rprods:
    [robj.kdp_to_r(rattc[k1].vars[kdpr], mlyr=rmlyr[k1],
                   a=RPARAMS[k1]['rkdp_a'], b=RPARAMS[k1]['rkdp_b'],
                   beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_z' in rprods:
    [robj.z_to_r(rattc[k1].vars[zh_r[robj.site_name]], mlyr=rmlyr[k1],
                 a=RPARAMS[k1]['rz_a'], b=RPARAMS[k1]['rz_b'],
                 beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
# =============================================================================
# Hybrid estimators
# =============================================================================
if 'r_kdp_zdr' in rprods:
    [robj.kdp_zdr_to_r(rattc[k1].vars[kdpr], rattc[k1].vars[zdrr],
                       mlyr=rmlyr[k1], a=RPARAMS[k1]['rkdpzdr_a'],
                       b=RPARAMS[k1]['rkdpzdr_b'], c=RPARAMS[k1]['rkdpzdr_c'],
                       beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_z_ah' in rprods:
    [robj.z_ah_to_r(rattc[k1].vars[zh_r[robj.site_name]],
                    rattc[k1].vars[ahr], mlyr=rmlyr[k1],
                    rz_a=RPARAMS[k1]['rz_a'], rz_b=RPARAMS[k1]['rz_b'],
                    # rz_a=(1/0.026)**(1/0.69), rz_b=1/0.69,
                    rband=rbands[robj.site_name], temp=temp, z_thld=z_thld,
                    beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_z_kdp' in rprods:
    [robj.z_kdp_to_r(rattc[k1].vars[zh_r[robj.site_name]],
                     rattc[k1].vars[kdpr], z_thld=z_thld, mlyr=rmlyr[k1],
                     rz_a=RPARAMS[k1]['rz_a'], rz_b=RPARAMS[k1]['rz_b'],
                     rkdp_a=RPARAMS[k1]['rkdp_a'],
                     rkdp_b=RPARAMS[k1]['rkdp_b'],
                     beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_z_zdr' in rprods:
    [robj.z_zdr_to_r(rattc[k1].vars[zh_r[robj.site_name]],
                     rattc[k1].vars[zdrr], mlyr=rmlyr[k1],
                     a=RPARAMS[k1]['rzhzdr_a'], b=RPARAMS[k1]['rzhzdr_b'],
                     c=RPARAMS[k1]['rzhzdr_c'],
                     beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]
if 'r_ah_kdp' in rprods:
    [robj.ah_kdp_to_r(
        rattc[k1].vars[zh_r[robj.site_name]], rattc[k1].vars[ahr],
        rattc[k1].vars[kdpr], mlyr=rmlyr[k1], rband=rbands[robj.site_name],
        temp=temp, z_thld=z_thld,
        rkdp_a=RPARAMS[k1]['rkdp_a'], rkdp_b=RPARAMS[k1]['rkdp_b'],
        # rah_a=rah_a, rah_b=rah_b, rkdp_a=rkdp_a, rkdp_b=rkdp_b,
        beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe.items()]

# =============================================================================
# Adaptive estimators
# =============================================================================
rqpe_opt = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in rattc.items()}
rqpe_opt2 = {k1: tp.qpe.qpe_algs.RadarQPE(robj)
             for k1, robj in rattc.items()}
if 'r_kdpopt' in rprods:
    rkdp_fit = {k1: tpx.rkdp_opt(
        rattc[k1].vars[kdp_kdpo[robj.site_name]],
        rattc[k1].vars[zh_kdpo[robj.site_name]],
        rkdp_stv=(RPARAMS[k1]['rkdp_a'], RPARAMS[k1]['rkdp_b']),
        zh_thr=((40, 50) if rbands[robj.site_name] == 'X' else (44.5, 45.5)),
        kdpmed=0.5, mlyr=rmlyr[k1], rband=rbands[robj.site_name],
        ptitle_pfx=f'{robj.elev_angle}'
        + f'{(" deg." if rbands[robj.site_name] == "X" else "")}'
        + f' -- {robj.scandatetime.strftime("%Y-%m-%d %H:%M:%S")}',
        plot_method=True)
                for k1, robj in rqpe_opt.items()}
    [robj.kdp_to_r(rattc[k1].vars[kdpr], mlyr=rmlyr[k1],
                   a=rkdp_fit[k1][0], b=rkdp_fit[k1][1],
                   beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_opt.items()]
    for k1, robjz in rqpe.items():
        robjz.r_kdpopt = rqpe_opt[k1].r_kdp
if 'r_zopt' in rprods:
    rzh_fit = {k1: tpx.rzh_opt(
        rattc[k1].vars[zh_zho[robj.site_name]], rqpe[k1].r_ah,
        rattc[k1].vars['AH [dB/km]'], pia=rattc[k1].vars['PIA [dB]'],
        mlyr=rmlyr[k1], minpia=(0.1), fit_ab=False,
        maxpia=(100 if rbands[robj.site_name] == 'X' else 100),
        rzfit_b=(2.14 if rbands[robj.site_name] == 'X' else 1.6),
        rz_stv=[[RPARAMS[k1]['rz_a'], RPARAMS[k1]['rz_b']],
                [r_coeffs[k1]['rz_a'], r_coeffs[k1]['rz_b']]],
        rz_stv_names={'trah': 'Diederich et al., 2015a',
                      'trzh': ['Diederich et al., 2015b'
                               if rbands[robj.site_name] == 'X' else
                               'Chen et al., 2021',
                               'Chen et al., 2023']},
        figlims_ah=((0, 3) if rbands[robj.site_name] == 'X' else (0, 0.5)),
        plot_method=PLOT_METHODS, plot_type='log',
        # ptitle_pfx=f'{robj.elev_angle}'
        # + f'{(" deg." if rbands[robj.site_name] == "X" else "")}'
        # + f' -- {robj.scandatetime.strftime("%Y-%m-%d %H:%M:%S")}'
        )
        for k1, robj in rqpe_opt.items()}
    [robj.z_to_r(rattc[k1].vars[zh_r[robj.site_name]], mlyr=rmlyr[k1],
                 a=rzh_fit[k1][0], b=rzh_fit[k1][1],
                 beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_opt.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zopt = rqpe_opt[k1].r_z
if 'r_zopt_ah' in rprods and 'r_zopt' in rprods:
    [robj.z_ah_to_r(rattc[k1].vars[zh_r[robj.site_name]],
                    rattc[k1].vars[ahr],
                    rz_a=rzh_fit[k1][0], rz_b=rzh_fit[k1][1],
                    rband=rbands[robj.site_name], temp=temp, z_thld=z_thld,
                    mlyr=rmlyr[k1],
                    beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_opt.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zopt_ah = rqpe_opt[k1].r_z_ah
if 'r_zopt_kdp' in rprods and 'r_zopt' in rprods:
    [robj.z_kdp_to_r(rattc[k1].vars[zh_r[robj.site_name]],
                     rattc[k1].vars[kdpr], z_thld=z_thld, mlyr=rmlyr[k1],
                     rz_a=rzh_fit[k1][0], rz_b=rzh_fit[k1][1],
                     rkdp_a=RPARAMS[k1]['rkdp_a'],
                     rkdp_b=RPARAMS[k1]['rkdp_b'],
                     beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_opt.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zopt_kdp = rqpe_opt[k1].r_z_kdp
if ('r_zopt_kdpopt' in rprods and 'r_zopt' in rprods
        and 'r_kdpopt' in rprods):
    [robj.z_kdp_to_r(rattc[cnt].vars[zh_r[robj.site_name]],
                     rattc[cnt].vars[kdpr], z_thld=z_thld, mlyr=rmlyr[cnt],
                     rz_a=rzh_fit[cnt][0], rz_b=rzh_fit[cnt][1],
                     rkdp_a=rkdp_fit[cnt][0], rkdp_b=rkdp_fit[cnt][1],
                     beam_height=rdata[cnt].georef['beam_height [km]'])
     for cnt, robj in rqpe_opt2.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zopt_kdpopt = rqpe_opt2[k1].r_z_kdp
if 'r_ah_kdpopt' in rprods:
    [robj.ah_kdp_to_r(
        rattc[k1].vars[zh_r[robj.site_name]], rattc[k1].vars[ahr],
        rattc[k1].vars[kdpr], mlyr=rmlyr[k1], rband=rbands[robj.site_name],
        temp=temp, z_thld=z_thld,
        rkdp_a=rkdp_fit[k1][0], rkdp_b=rkdp_fit[k1][1],
        beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_opt.items()]
    for k1, robjz in rqpe.items():
        robjz.r_ah_kdpopt = rqpe_opt[k1].r_ah_kdp
# =============================================================================
# Estimators using non-synthetic variables
# =============================================================================
rqpe_nfc = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in rattc.items()}
if 'r_kdpo' in rprods:
    kdpr2 = 'KDP* [deg/km]'  # Vulpiani
    [robj.kdp_to_r(rattc[k1].vars[kdpr2], mlyr=rmlyr[k1],
                   a=r_coeffs[robj.site_name]['rkdp_a'],
                   b=r_coeffs[robj.site_name]['rkdp_b'],
                   beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_nfc.items()]
    for k1, robjk in rqpe.items():
        robjk.r_kdpo = rqpe_nfc[k1].r_kdp
if 'r_zo' in rprods:
    zhr2 = 'ZH [dBZ]'  # ZHattc
    [robj.z_to_r(rattc[k1].vars[zhr2], mlyr=rmlyr[k1],
                 a=r_coeffs[robj.site_name]['rz_a'],
                 b=r_coeffs[robj.site_name]['rz_b'],
                 beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_nfc.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zo = rqpe_nfc[k1].r_z
if 'r_zo_ah' in rprods:
    [robj.z_ah_to_r(rattc[k1].vars[zhr2], rattc[k1].vars[ahr],
                    mlyr=rmlyr[k1], temp=temp, z_thld=z_thld,
                    rband=rbands[robj.site_name],
                    rz_a=r_coeffs[robj.site_name]['rz_a'],
                    rz_b=r_coeffs[robj.site_name]['rz_b'],
                    rah_a=r_coeffs[robj.site_name]['rah_a'],
                    rah_b=r_coeffs[robj.site_name]['rah_b'],
                    beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_nfc.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zo_ah = rqpe_nfc[k1].r_z_ah
if 'r_zo_kdp' in rprods:
    [robj.z_kdp_to_r(rattc[k1].vars[zhr2], rattc[k1].vars[kdpr2],
                     rz_a=r_coeffs[robj.site_name]['rz_a'],
                     rz_b=r_coeffs[robj.site_name]['rz_b'],
                     rkdp_a=r_coeffs[robj.site_name]['rkdp_a'],
                     rkdp_b=r_coeffs[robj.site_name]['rkdp_b'],
                     z_thld=z_thld, mlyr=rmlyr[k1],
                     beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_nfc.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zo_kdp = rqpe_nfc[k1].r_z_kdp
if 'r_zo_zdr' in rprods:
    [robj.z_zdr_to_r(rattc[k1].vars[zhr2], rattc[k1].vars[zdrr],
                     mlyr=rmlyr[k1],
                     a=RPARAMS[k1]['rzhzdr_a'], b=RPARAMS[k1]['rzhzdr_b'],
                     c=RPARAMS[k1]['rzhzdr_c'],
                     beam_height=rdata[k1].georef['beam_height [km]'])
     for k1, robj in rqpe_nfc.items()]
    for k1, robjz in rqpe.items():
        robjz.r_zo_zdr = rqpe_nfc[k1].r_z_zdr
if 'r_aho_kdpo' in rprods:
    [robj.ah_kdp_to_r(
        rattc[k1].vars[zhr2], rattc[k1].vars[ahr], rattc[k1].vars[kdpr2],
        mlyr=rmlyr[k1], temp=temp, z_thld=z_thld, rband=rbands[robj.site_name],
        rah_a=r_coeffs[robj.site_name]['rah_a'],
        rah_b=r_coeffs[robj.site_name]['rah_b'],
        rkdp_a=r_coeffs[robj.site_name]['rkdp_a'],
        rkdp_b=r_coeffs[robj.site_name]['rkdp_b'],
        beam_height=rdata[k1].georef['beam_height [km]'])
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
    # max_rkm = 151
# =============================================================================
# RZ relation is modified by applying a factor to data within the ML.
rqpe_ml = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in rattc.items()}
[robj.z_to_r(rattc[k1].vars[zh_r[robj.site_name]],
             a=RPARAMS[k1]['rz_a'], b=RPARAMS[k1]['rz_b'])
 for k1, robj in rqpe_ml.items()]
for k1, robj in rqpe_ml.items():
    robj.r_z['Rainfall [mm/h]'] = np.where(
        (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 2)
        & (rattc[k1].vars[zh_r[robj.site_name]] > thr_zwsnw),
        robj.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)
# =============================================================================
# RZ relation is modified by applying a factor to data above the ML.
rqpe_aml = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in rattc.items()}
[robj.z_to_r(rattc[k1].vars[zh_r[robj.site_name]],
             a=RPARAMS[k1]['rz_a'], b=RPARAMS[k1]['rz_b'])
 for k1, robj in rqpe_aml.items()]
for k1, robj in rqpe_aml.items():
    robj.r_z['Rainfall [mm/h]'] = np.where(
        (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 3.),
        robj.r_z['Rainfall [mm/h]']*f_rz_aml,
        rqpe_ml[k1].r_z['Rainfall [mm/h]'])
# Correct all other variables
# if qpe_amlb:
for k1, robj in rqpe.items():
    [setattr(robj, rp, {(k2): (np.where(
        (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1),
        getattr(robj, rp)['Rainfall [mm/h]'],
        rqpe_aml[k1].r_z['Rainfall [mm/h]']) if 'Rainfall' in k2 else v1)
        for k2, v1 in getattr(robj, rp).items()})
        for rp in robj.__dict__.keys() if rp.startswith('r_')]
# =============================================================================
# rz_hail is applied to data below the ML with Z > 55 dBZ
rqpe_hail = {k1: tp.qpe.qpe_algs.RadarQPE(robj) for k1, robj in rattc.items()}
[robj.z_to_r(rattc[k1].vars[zh_r[robj.site_name]],
             a=RPARAMS[k1]['rz_haila'], b=RPARAMS[k1]['rz_hailb'])
 for k1, robj in rqpe_hail.items()]
for k1, robj in rqpe_hail.items():
    robj.r_z['Rainfall [mm/h]'] = np.where(
        (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1)
        & (rattc[k1].vars[zh_r[robj.site_name]] >= thr_zhail),
        robj.r_z['Rainfall [mm/h]'], np.nan)
# Set a limit in range
# grid_range = [np.ones_like(robj.georef['beam_height [km]'])
#               * robj.georef['range [m]']/1000 for robj in rattc]
# for rcnt, robj in enumerate(rqpe_hail):
#     robj.r_z['Rainfall [mm/h]'] = np.where(
#         (rmlyr[rcnt].mlyr_limits['pcp_region [HC]'] == 3.)
#         & (grid_range[rcnt] > max_rkm), 0, robj.r_z['Rainfall [mm/h]'])
#     robj.r_z['Rainfall [mm/h]'] = np.where(
#         (np.isnan(rqpe[rcnt].r_z['Rainfall [mm/h]'])), np.nan,
#         robj.r_z['Rainfall [mm/h]'])
# # Correct all other variables
for k1, robj in rqpe.items():
    [setattr(
        robj, rp, {(k2): (np.where(
            (rmlyr[k1].mlyr_limits['pcp_region [HC]'] == 1)
            & (rattc[k1].vars[zh_r[robj.site_name]] >= thr_zhail),
            rqpe_hail[k1].r_z['Rainfall [mm/h]'],
            getattr(robj, rp)['Rainfall [mm/h]']) if 'Rainfall' in k2
            else v1) for k2, v1 in getattr(robj, rp).items()})
        for rp in robj.__dict__.keys() if rp.startswith('r_')]

toc1 = perf_counter()
print(f'TIME ELAPSED [Quality-control]: {dt.timedelta(seconds=toc1-tic)}')

# %%
# PLOT_METHODS = True
# v2p = 'AH [dB/km]'
v2p = 'ZH+ [dBZ]'
v2p = 'beam_height [km]'
# v2p = 'PhiDP [deg]'
# v2p = 'rhoHV [-]'
# v2p = 'ZDR [dB]'
# v2p = 'KDP* [deg/km]'
if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(
        robj.georef, robj.params, robj.georef, var2plot=v2p,
        # robj.georef, robj.params, rattc[k1].vars, var2plot=v2p,
        cpy_feats={'status': True, 'tiles': False, 'alpha_rad': 0.9,
                   'alpha_tiles': 1, 'tiles_source': 'OSM',
                   'tiles_style': None}, data_proj=ccrs.PlateCarree(),
        # rd_maxrange=True, ring=[1, 5],
        proj_suffix='wgs84', fig_size=fig_size, xlims=xlims, ylims=ylims,
        vars_bounds={'Beam_height [km]': [0, 5, int(1+((5-0)/0.2))]},)
     for k1, robj in rdata.items()]
    [tp.datavis.rad_display.plot_ppi(
        rdata[k1].georef, rdata[k1].params, robj.vars, var2plot=v2p,
        vars_bounds={'KDP [deg/km]': (-1, 3, 17)})
        for k1, robj in rattc.items()]

# %%
v2p = 'Rainfall [mm/h]'
if PLOT_METHODS:
    [tp.datavis.rad_display.plot_ppi(
        robj.georef, robj.params, rqpe[k1].r_kdpopt, xlims=xlims, ylims=ylims,
        cpy_feats={'status': True}, data_proj=ccrs.UTM(zone=32), var2plot=v2p,
        proj_suffix='utm', fig_size=fig_size)
     for k1, robj in rdata.items()]
    [tp.datavis.rad_display.plot_ppi(
        robj.georef, robj.params, rqpe[k1].r_kdpopt, var2plot=v2p)
        for k1, robj in rdata.items()]
    tp.datavis.rad_display.plot_mgrid(
        [i.georef for k, i in rdata.items()],
        [i.params for k, i in rdata.items()],
        [i.r_zopt for k, i in rqpe.items()], cpy_feats={'status': True},
        var2plot=v2p, proj_suffix='utm', data_proj=ccrs.UTM(zone=32),
        xlims=xlims, ylims=ylims)

# %%
# =============================================================================
# adds altitude to BH for analysis
# =============================================================================
# for k1, robj in rdata.items():
#     print(robj.params['altitude [m]']/1000)
#     robj.georef['beam_height [km]'] += robj.params['altitude [m]']/1000
#     # robj.georef['beam_height [km]'] -= robj.params['altitude [m]']/1000

# %%
nbins = max([robj.params['ngates'] for k1, robj in rdata.items()])
nbins = 500
# nbins = 1000

if COMPOSITE:
    tic = perf_counter()
    for k1, robj in rqpe.items():
        [setattr(
            robj, rp, {(k2): (
                np.where(getattr(robj, rp)['Rainfall [mm/h]'] == 0, np.nan,
                         getattr(robj, rp)['Rainfall [mm/h]'])
                if 'Rainfall' in k2 else v1)
                for k2, v1 in getattr(robj, rp).items()})
            for rp in robj.__dict__.keys() if rp.startswith('r_')]
    # derive UTM Zone 29 coordinates of range-bin centroids
    # create osr projection using epsg number for UTM Zone 29
    rads_coord = {k1: np.array([robj.georef['grid_utmx'].flatten(),
                  robj.georef['grid_utmy'].flatten()]).T
                  for k1, robj in rdata.items()}
    # define target grid for composition
    xmin, xmax, ymin, ymax = tpx.bbox(
        np.concatenate([i for k, i in rads_coord.items()], axis=0))
    # x = np.linspace(xmin, xmax + binres, int(nbins))
    # y = np.linspace(ymin, ymax + binres, int(nbins))
    cgrid_x = np.linspace(xmin - nbins, xmax + nbins, int(nbins*2))
    cgrid_y = np.linspace(ymin - nbins, ymax + nbins, int(nbins*2))
    # x = np.linspace(xmin, xmax, int(nbins*2))
    # y = np.linspace(ymin, ymax, int(nbins*2))
    grid_coords = wrl.util.gridaspoints(cgrid_y, cgrid_x)
    # derive quality information - in this case, the pulse volume
    pulse_volumes = {k1: np.tile(wrl.qual.pulse_volume(
        robj.georef['range [m]'], robj.params['gateres [m]'],
        robj.params['beamwidth [deg]']), robj.params['nrays'])
        for k1, robj in rdata.items()}
    pulse_volumes2 = {k1:
                      {'pulse_volume [exp]':
                       pulse_volumes[k1].reshape(
                           rdata[k1].georef['beam_height [km]'].shape)}
                         for k1, robj in pulse_volumes.items()}
    # interpolate polar radar-data and quality data to the grid
    rd_quality_gridded = [wrl.comp.togrid(
        robj, grid_coords, rdata[k1].georef['range [m]'].max()
        + rdata[k1].georef['range [m]'][0], robj.mean(axis=0),
        pulse_volumes[k1], wrl.ipol.Nearest)
        for k1, robj in rads_coord.items()]
    # Define the radar rainfall products to be composed.
    rfields = {k1: {k: v for k, v in robj.__dict__.items()
                    if isinstance(v, dict)} for k1, robj in rqpe.items()}
    # rfieldk = set([k for i in rfields for k in i.keys()])
    rd_gridded = {k1: [wrl.comp.togrid(
        robj, grid_coords, rdata[k2].georef['range [m]'].max()
        + rdata[k2].georef['range [m]'][0], robj.mean(axis=0),
        rfields[k2][k1]['Rainfall [mm/h]'][:, 0:].ravel(), wrl.ipol.Nearest)
        for k2, robj in rads_coord.items()] for k1 in rprods}
    # compose the both radar-data based on the quality information
    # calculated above
    ires = 0.001
    # ires = 0
    rcomps = {k1: wrl.comp.compose_weighted(
        v1, [1. / (i + ires) for i in rd_quality_gridded])
        for k1, v1 in rd_gridded.items()}
    rcomps = {k1: np.ma.masked_invalid(v1) for k1, v1 in rcomps.items()}
    rcomps = {k1: v1.reshape((len(cgrid_x), len(cgrid_y)))
              for k1, v1 in rcomps.items()}
    # TODO: add option to compose polarimetric moments, BH and MLH.
    # =============================================================================
    # Only valid when using radars operating at the same frequency/band
    # =============================================================================
    # FOR BEAM HEIGHT ONLY
    rd_quality_gridded2 = [wrl.comp.togrid(
        robj, grid_coords, rdata[k1].georef['range [m]'].max()
        + rdata[k1].georef['range [m]'][0], robj.mean(axis=0),
        # pulse_volumes[k1], wrl.ipol.Idw)
        pulse_volumes[k1], wrl.ipol.Nearest)
        for k1, robj in rads_coord.items()]
    rd_gridded2 = [wrl.comp.togrid(
        robj, grid_coords, rdata[k1].georef['range [m]'].max()
        + rdata[k1].georef['range [m]'][0], robj.mean(axis=0),
        rdata[k1].georef['beam_height [km]'][:, 0:].ravel(),
        # rmlyr[k1].mlyr_limits['pcp_region [HC]'][:, 0:].ravel(),
        # wrl.ipol.Idw) for k1, robj in rads_coord.items()]
        wrl.ipol.Nearest) for k1, robj in rads_coord.items()]
    rcomp_ml = wrl.comp.compose_ko(
        rd_gridded2, [1. / (i + ires) for i in rd_quality_gridded2])
    rcomp_ml = np.ma.masked_invalid(rcomp_ml)
    rcomp_ml = {'beam_height [km]':
                rcomp_ml.reshape((len(cgrid_x), len(cgrid_y)))}
    # rvars = ['ADP [dB/km]', 'AH [dB/km]', 'KDP [deg/km]',
    #          'ZDR [dB]', 'ZH [dBZ]']
    # rvfields = [{k1: v1 for k1, v1 in robj.vars.items() if k1 in rvars}
    #             for cnt, robj in enumerate(rattc)]
    # rd_gridded3 = {k1: [wrl.comp.togrid(
    #     robj, grid_coords, rdata[cnt].georef['range [m]'].max()
    #     + rdata[cnt].georef['range [m]'][0], robj.mean(axis=0),
    #     rvfields[cnt][k1][:, 0:].ravel(), wrl.ipol.Nearest)
    #     for cnt, robj in enumerate(rads_coord)] for k1 in rvars}
    # rcomps_pv = {k1: wrl.comp.compose_weighted(
    #     v1, [1. / (i + 0.001) for i in rd_quality_gridded])
    #     for k1, v1 in rd_gridded3.items()}
    # rcomps_pv = {k1: np.ma.masked_invalid(v1)
    #              for k1, v1 in rcomps_pv.items()}
    # rcomps_pv = {k1: v1.reshape((len(x), len(y)))
    #              for k1, v1 in rcomps_pv.items()}
    # [robj.z_to_r(rzhah[cnt].vars['ZH [dBZ]'], mlyr=rmlyr[cnt],
    #              a=RPARAMS[k1]['rz_a'], b=RPARAMS[k1]['rz_b'],
    #              beam_height=rdata[cnt].georef['beam_height [km]'])
    #  if 'xpol' in robj.site_name.lower() else
    #  robj.z_to_r(rattc[cnt].vars['ZH [dBZ]'], mlyr=rmlyr[cnt],
    #              a=RPARAMS[k1]['rz_a'], b=RPARAMS[k1]['rz_b'],
    #              beam_height=rdata[cnt].georef['beam_height [km]'])
    #  for cnt, robj in enumerate(rqpe)]
    rcomp_params = copy.copy(rdata[list(rdata)[0]].params)
    dt_mean1 = [robj.scandatetime for k1, robj in rqpe.items()]
    dt_mean = min(dt_mean1)+(max(dt_mean1)-min(dt_mean1))/2
    # rqpe_dt.append(dt_mean)
    rcomps['datetime'] = dt_mean
    rcomps['elev_ang [deg]'] = {robj.site_name: robj.elev_angle
                                for k1, robj in rqpe.items()}
    rcomp_georef = {
        'grid_utmx':
            grid_coords[:, 0].reshape((len(cgrid_x), len(cgrid_y))),
        'grid_utmy':
            grid_coords[:, 1].reshape((len(cgrid_x), len(cgrid_y)))}
    epsg_to_osr = 32632
    wgs84 = wrl.georef.get_default_projection()
    utm = wrl.georef.epsg_to_osr(epsg_to_osr)
    rcomp_georef['grid_wgs84x'], rcomp_georef['grid_wgs84y'] = (
        wrl.georef.reproject(rcomp_georef['grid_utmx'],
                             rcomp_georef['grid_utmy'],
                             # projection_source=utm,
                             # projection_target=wgs84)
                             src_crs=utm, trg_crs=wgs84)
        )
    toc2 = perf_counter()
    print(f'TIME ELAPSED [Composite]: {dt.timedelta(seconds=toc2-tic)}')
# %%
if COMPOSITE:
    # xlims, ylims = (6.6, 7.2), (50.6, 51.2)
    # ralg = 'r_ah'
    # ralg = 'r_kdpopt'
    # ralg = 'r_zopt'
    ralg = 'Beam height'
    rcomp_vars = rcomp_ml
    # rcomp_vars = {'ZH [dBZ]': composite}
    # rcomp_vars = {'Rainfall [mm/h]': rcomps[ralg]}
    rad_display.plot_ppi(
        rcomp_georef, rcomp_params, rcomp_vars,  # var2plot='Rainfall [mm/h]',
        cpy_feats={'status': True, 'tiles': True, 'alpha_rad': 0.9,
                   'alpha_tiles': 1, 'tiles_source': 'OSM',
                   'tiles_style': None},
        proj_suffix='wgs84',
        data_proj=ccrs.PlateCarree(), xlims=xlims, ylims=ylims,
        fig_size=fig_size,
        # ucmap='jet',
        # ucmap='terrain',
        # ucmap='gist_earth_r',
        # vars_bounds={'Beam_height [km]': [0, 5, int(1+((5-0)/0.25))]},
        vars_bounds={'Beam_height [km]': [0, 5, int(1+((5-0)/0.2))]},
        fig_title=(f'Radar Composite [{ralg}]: '
                   + f"{rcomp_params['datetime']:%Y-%m-%d %H:%M}"))
    # tp.datavis.rad_display.plot_mgrid(
    #     [i.georef for k, i in rdata.items()],
    #     [i.params for k, i in rdata.items()],
    #     [i for k, i in pulse_volumes2.items()], cpy_feats={'status': True},
    #     proj_suffix='utm', data_proj=ccrs.UTM(zone=32),
    #     var2plot='pulse_volume [exp]', xlims=xlims, ylims=ylims,
    #     vars_bounds={'pulse_volume [exp]': [0, 1.6e9, 17]}, ucmap='viridis')
# %%
# =============================================================================
# Read RG data
# =============================================================================
# RG_WDIR = (LWDIR + 'pd_rdres/dwd_rg/')
# DWDRG_MDFN = (RG_WDIR + 'RR_Stundenwerte_Beschreibung_Stationen2024.csv')
# RG_NCDATA = (LWDIR + 'pd_rdres/dwd_rg/'
#              # + 'nrw_20210713_20210715_1h_1hac/'
#              + 'nrw_20170723_20170727_1h_1hac/'
#              )

# =============================================================================
# Init raingauge object
# =============================================================================
# rg_data = tpx.RainGauge(RG_WDIR, nwk_opr='DWD')

# =============================================================================
# Read metadata of all DWD rain gauges (location, records, etc)
# =============================================================================
# rg_data.get_dwdstn_mdata(DWDRG_MDFN, plot_methods=False)

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
# bbox_xlims, bbox_ylims = (6, 11.), (48.6, 52.8)  # DWDXPOL
# bbox_xlims, bbox_ylims = (6, 10.7), (49, 52.6)  # DWDXPOLF
# # 4.59351 , 4.19004, 10.8476, 11.0174
# bbox_xlims = (rcomp_georef['grid_wgs84x'].min(),
#               rcomp_georef['grid_wgs84x'].max())
# bbox_ylims = (rcomp_georef['grid_wgs84y'].min(),
#               rcomp_georef['grid_wgs84y'].max())
# rg_data.get_stns_box(rg_data.dwd_stn_mdata, bbox_xlims=bbox_xlims,
#                      bbox_ylims=bbox_ylims, plot_methods=True,
#                      surface=rcomp_georef)

# a = tpx.bbox(rcomp_georef['grid_wgs84x'], rcomp_georef['grid_wgs84y'])

# =============================================================================
# Download DWD rg data
# =============================================================================
# for hour in range(96):
#     start_time = dt.datetime(2017, 7, 23, 0, 0, 0)
#     # print(hour)
#     start_time = start_time + dt.timedelta(hours=hour)
#     print(start_time)
#     stop_time = start_time + dt.timedelta(hours=1)
#     print(stop_time)
#     # start_time = start_time + datetime.timedelta(hours=hour+1)
#     # print(start_time)
#     # for station_id in rg_data.stn_near_rad['stations_id']:
#     for station_id in rg_data.stn_bbox['station_id']:
#         rg_data.get_dwdstn_nc(station_id, start_time, stop_time,
#                               dir_ncdf=RG_NCDATA)

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

# %%
# =============================================================================
# FINAL OBJ
# =============================================================================
rd_qcatc = {}
for k1, robj in rdata.items():
    rd_qcat = tp.attc.attc_zhzdr.AttenuationCorrection(robj)
    rd_qcat.georef = robj.georef
    rd_qcat.params = robj.params
    rd_qcat.alpha_ah = np.nanmean([np.nanmean(i)
                                   for i in rattc[k1].vars['alpha [-]']])
    rd_qcat.phidp0 = ropdp[k1].phidp_offset
    # rd_qcat.vars = dict(rattc[k1].vars)
    # del rd_qcat.vars['alpha [-]']
    # # del rd_qcat.vars['ADP [dB/km]']
    # # del rd_qcat.vars['ZH [dBZ]']
    # del rd_qcat.vars['ZH* [dBZ]']
    # del rd_qcat.vars['beta [-]']
    # # del rd_qcat.vars['PIA [dB]']
    # # del rd_qcat.vars['KDP [deg/km]']
    # # del rd_qcat.vars['KDP* [deg/km]']
    # # del rd_qcat.vars['PhiDP [deg]']
    # del rd_qcat.vars['PhiDP* [deg]']
    # rd_qcat.vars['Rainfall [mm/h]'] = rqpe[k1].r_zopt['Rainfall [mm/h]']
    rd_qcat.vars = {}
    rd_qcat.vars['ZH [dBZ]'] = rattc[k1].vars['ZH+ [dBZ]']
    rd_qcat.vars['ZDR [dB]'] = rattc[k1].vars['ZDR [dB]']
    rd_qcat.vars['PhiDP [deg]'] = rattc[k1].vars['PhiDP [deg]']
    rd_qcat.vars['rhoHV [-]'] = rozdr[k1].vars['rhoHV [-]']
    rd_qcat.vars['AH [dB/km]'] = rattc[k1].vars['AH [dB/km]']
    rd_qcat.vars['KDP [deg/km]'] = rattc[k1].vars['KDP+ [deg/km]']
    # rd_qcat.vars['ZH+ [dBZ]'] = rattc[k1].vars['ZH+ [dBZ]']
    # rd_qcat.vars['Beam_height [km]'] = rdata[k1].georef['beam_height [km]']
    rd_qcatc[k1] = rd_qcat

# if PLOT_METHODS:
[tp.datavis.rad_display.plot_setppi(
    rdata[k1].georef, rdata[k1].params, rd_qcatc[k1].vars,
    # vars_bounds={'AH [dB/km]': [0, 0.5, 17], 'KDP [deg/km]': [-1, 3, 17]}
    )
for k1, robj in rdata.items()]

# %%
# =============================================================================
# PLOTS
# =============================================================================
# bh = [i.georef for i in rdata]
# bh2 = [{k: v for k, v in i.items() if 'beam' in k} for i in bh]

v2p = 'ZH+ [dBZ]'
# v2p = 'PhiDP [deg]'
# v2p = 'rhoHV [-]'
# v2p = 'ZDR [dB]'
# v2p = 'KDP+ [deg/km]'

# [tp.datavis.rad_display.plot_ppi(rdata[k1].georef, rdata[k1].params,
#                                  robj.vars)
#  for k1, robj in rdata.items()]

# if PLOT_METHODS:
nplot = list(rdata)[-1]
tp.datavis.rad_interactive.ppi_base(
    rdata[nplot].georef, rdata[nplot].params,
    # rdata[nplot].vars,
    # ropdp[nplot].vars,
    # rsnr[nplot].vars,
    # rnme[nplot].vars,
    # rozdr[nplot].vars,
    # rattc[nplot].vars,
    rd_qcatc[nplot].vars,
    # bh2[nplot],
    # rzhah[nplot].vars,
    # coord='rect',
    # var2plot='rhoHV [-]',
    # var2plot='beta',
    # var2plot='PIA [dB]',
    # var2plot='AH [dB/km]',
    # var2plot='KDP+ [deg/km]',
    # var2plot='V [m/s]',
    # var2plot='Rainfall [mm/h]',
    # var2plot='PhiDP [deg]',
    # var2plot='beam_height [km]',
    # var2plot='ZDR [dB]',
    # var2plot='ZH+ [dBZ]',
    # ylims={'ZH [dBZ]': (0, 50)},
    # radial_xlims=(45, 65),
    vars_bounds={'KDP [deg/km]': (-1, 3, 17),
                 'PIA [dB]': (0, 19, 20),
                 'AH [dB/km]': (0, 0.02, 13),
                 # 'PhiDP [deg]': (-10, 85, 20),
                 # 'ZH [dBZ]': [5, 50, 10],
                 # 'rhoHV [-]': (0.3, .9, 1),
                 # 'Rainfall [mm/h]': [0, 55, 12],
                 },
    # ucmap='tpylsc_useq_model',
    # ucmap='turbo',
    mlyr=rmlyr[nplot]
    )
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
# %%
# import matplotlib.colors as mcolors
# import matplotlib as mpl

# fvars = {}
# # cmap = mpl.colormaps['tpylsc_dbu_rd']
# fvars['ZH [dBZ]'] = (np.nan_to_num(rd_gridded[1].reshape((len(x), len(y))))
#                      - np.nan_to_num(rd_gridded[2].reshape((len(x), len(y)))))
# rcomp_georef = {}
# # rcomp_georef = {'xgrid': x, 'ygrid': y}
# rcomp_georef = {'grid_utmx': x, 'grid_utmy': y}
# tp.datavis.rad_display.plot_ppi(rcomp_georef, rcomp_params, fvars,
#                                 cpy_feats={'status': True},
#                                 data_proj=ccrs.UTM(zone=32),
#                                 proj_suffix='utm',
#                                 fig_size=(11.5, 7),
#                                 xlims=xlims, ylims=ylims,
#                                 # var2plot='ZH* [dBZ]',
#                                 vars_bounds={'ZH [dBZ]': (-10, 10, 19)},
#                                 # unorm=unorm,
#                                 # ucmap='tpylsc_rd_w_k_r',
#                                 ucmap='tpylsc_div_rd_w_k_r'
#                                 # ucmap='tpylsc_grey',
#                                )
