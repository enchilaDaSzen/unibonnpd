#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:55:05 2023

@author: dsanchez
"""

import datetime as dt
from time import perf_counter
import os
import pickle
import sys
import numpy as np
import towerpy as tp
import wradlib as wrl
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi
LWDIR = '/home/dsanchez/sciebo_dsr/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
sys.path.append(LWDIR + 'codes/github/unibonnpd/')
from radar.rparams_dwdxpol import RPARAMS
from radar import twpext as tpx

# =============================================================================
# Define working directory, and date-time
# =============================================================================
# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24hr [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24hr [NO JXP]
# START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24hr []
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2018, 12, 2, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 24hr [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 0, 0)  # 16 hr [NO BXP][JXP8am]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24 hr [NO BXP]
START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24 hr [NO BXP]

STOP_TIME = START_TIME+dt.timedelta(hours=24)

data4calib = 'qvps'

EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
PDIR = None
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'
# PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'

DIRPROFSCAL = LWDIR + f"pd_rdres/qvps_d4calib/{START_TIME.strftime('%Y%m%d')}/"
MFS_DIR = (LWDIR + 'codes/github/unibonnpd/radar/nme/xpol_mfs/')
CLM_DIR = (LWDIR + 'codes/github/unibonnpd/radar/nme/xpol_clm/')

# =============================================================================
# Define radar site
# =============================================================================
# Choose only one site at a time
# Boxpol, Juxpol, Essen, Flechtdorf, Neuheilenbach, Offenthal
RSITE = 'Offenthal'
RPARAMS = {RSITE: next(item for item in RPARAMS if item['site_name'] == RSITE)}

# =============================================================================
# Read-in QVPs data
# =============================================================================
RCAL_FILES = {RSITE: DIRPROFSCAL+n for n in sorted(os.listdir(DIRPROFSCAL))
              if data4calib in n and RPARAMS[RSITE]['site_name'] in n}

with open(RCAL_FILES[RSITE], 'rb') as breader:
    profs_data = pickle.load(breader)

mlyrhv = [i for i in profs_data['mlyr'] if ~np.isnan(i.ml_top)]
mlt_avg = np.nanmean([i.ml_top for i in profs_data['mlyr']])
mlk_avg = np.nanmean([i.ml_thickness for i in profs_data['mlyr']])
mlb_avg = np.nanmean([i.ml_bottom for i in profs_data['mlyr']])
phidpOv = [i for i in profs_data['phidpO'] if ~np.isnan(i.phidp_offset)]
zdrOv = [i for i in profs_data['zdrO'] if ~np.isnan(i.zdr_offset)]

# =============================================================================
# List PPI radar data
# =============================================================================
RSITE_FILES = {k: tpx.get_listfilesxpol(i['site_name'], START_TIME, STOP_TIME,
                                        i['elev'], parent_dir=PDIR)
               if 'xpol' in i['site_name'].lower() else
               tpx.get_listfilesdwd(i['site_name'], START_TIME, STOP_TIME,
                                    i['elev'], parent_dir=PDIR)
               for k, i in RPARAMS.items()}

# Set directory to save results
RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}/"
           + f"rsite_qc/{RPARAMS[RSITE]['site_name']}/")

# =============================================================================
# Set parameters related to QC
# =============================================================================
if mlb_avg > 1.5:
    temp = 15
else:
    temp = 5
    temp = 15
max_diffdtmin = 120

# ZH offset
if 'xpol' in RPARAMS[RSITE]['site_name']:
    # zh_off = RPARAMS[RSITE]['zh_offset'].get(START_TIME.strftime("%Y%m%d"))
    zh_off = 0
else:
    zh_off = 0
# PhiDP(0)
if RPARAMS[RSITE]['site_name'] == 'Juxpol' and START_TIME.year > 2018:
    RPARAMS[RSITE]['signpdp'] = -1
else:
    RPARAMS[RSITE]['signpdp'] = 1
if (RPARAMS[RSITE]['site_name'] == 'Boxpol'
        and RPARAMS[RSITE]['elev'] != 'n_ppi_010deg'):
    RPARAMS[RSITE]['bclass'] -= 64
preset_phidp = RPARAMS[RSITE]['phidp_prst'].get(START_TIME.strftime("%Y%m%d"))
preset_nlvl = RPARAMS[RSITE]['nlvl'].get(START_TIME.strftime("%Y%m%d"))
# modifies noise_level if zh_offset is corrected beforehand (JXP)
# preset_nlvl = (29-5, 33-5, 0.1)

# =============================================================================
# Set plotting parameters
# =============================================================================
PLOT_METHODS = False

# %%
# scan = RSITE_FILES[RSITE][211]
# scan = RSITE_FILES[RSITE][144]
# scan = RSITE_FILES[RSITE][0]

tic = perf_counter()

for scan in RSITE_FILES[RSITE]:
    # =====================================================================
    # Import data into towerpy using wradlib
    # =====================================================================
    rdata = tpx.Rad_scan(scan, RPARAMS[RSITE]['site_name'])
    try:
        if rdata.site_name.lower() == 'boxpol':
            rdata.ppi_xpol()
        elif rdata.site_name.lower() == 'juxpol':
            rdata.ppi_xpol(scan_elev=RPARAMS[RSITE]['elev'])
        else:
            rdata.ppi_dwd()
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                               rdata.vars)
        rband = RPARAMS[RSITE]['rband']
        # =============================================================================
        # Allocate closest dataset from QVPs (for ML, offsets, etc)
        # =============================================================================
        if mlyrhv:
            idx_mlh, mlh_dt = min(zip(range(
                len([i.scandatetime for i in mlyrhv])),
                    [i.scandatetime for i in mlyrhv]),
                key=lambda x: (x[1] > rdata.scandatetime,
                               abs(x[1] - rdata.scandatetime)))
        else:
            idx_mlh, mlh_dt = (0, profs_data['mlyr'][0].scandatetime)
        idx_phidp0, phidp0_dt = min(zip(range(
            len([i.scandatetime for i in phidpOv])),
                [i.scandatetime for i in phidpOv]),
            key=lambda x: (x[1] > rdata.scandatetime,
                           abs(x[1] - rdata.scandatetime)))
        idx_zdr0, zdr0_dt = min(zip(range(
            len([i.scandatetime for i in zdrOv])),
                [i.scandatetime for i in zdrOv]),
            key=lambda x: (x[1] > rdata.scandatetime,
                           abs(x[1] - rdata.scandatetime)))
        idx_pclass, pclass_dt = min(zip(range(
            len(profs_data['dtrs'])), profs_data['dtrs']),
            key=lambda x: (x[1] > rdata.scandatetime,
                           abs(x[1] - rdata.scandatetime)))
        # =============================================================================
        # ZH offset correction
        # =============================================================================
        rdata.vars['ZH [dBZ]'] += zh_off
        rdata.zh_offset = zh_off
        # =============================================================================
        # rhoHV noise-correction
        # =============================================================================
        rcrho = tpx.rhoHV_Noise_Bias(rdata)
        rcrho.iterate_radcst(
            rdata.georef, rdata.params, rdata.vars, noise_lvl=preset_nlvl,
            rhohv_theo=RPARAMS[RSITE]['rhvtc'], data2correct=rdata.vars)
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_ppidiff(
                rdata.georef, rdata.params, rdata.vars, rcrho.vars,
                var2plot1='rhoHV [-]', var2plot2='rhoHV [-]',
                ucmap_diff='tpylsc_div_dbu_rd', diff_lims=[-0.5, 0.5, .1])
        # =============================================================================
        # Noise suppression
        # =============================================================================
        if rdata.params['radar constant [dB]'] <= 0:
            min_snr = -rcrho.rhohv_corrs['Noise level [dB]']
        else:
            min_snr = rcrho.rhohv_corrs['Noise level [dB]']
        print(f"minSNR = {min_snr:.2f} dB")
        rsnr = tp.eclass.snr.SNR_Classif(rdata)
        rsnr.signalnoiseratio(
            rdata.georef, rdata.params, rcrho.vars,
            min_snr=min_snr,
            data2correct=rcrho.vars, plot_method=PLOT_METHODS)
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                               rsnr.vars)
        # =============================================================================
        # PhiDP quality control and processing
        # =============================================================================
        # Modify the PhiDP sign (only for JXP)
        rsnr.vars['PhiDP [deg]'] *= RPARAMS[RSITE]['signpdp']
        ropdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)
        ropdp.phidp_offset_qvps = phidpOv[idx_phidp0].phidp_offset
        ropdp.offsetdetection_ppi(rsnr.vars, preset=preset_phidp)
        # preset_phidp = ropdp.phidp_offset
        ropdp.offset_correction(rsnr.vars['PhiDP [deg]'],
                                phidp_offset=ropdp.phidp_offset,
                                data2correct=rsnr.vars)
        uphidp = np.ascontiguousarray(
            wrl.dp.unfold_phi(ropdp.vars['PhiDP [deg]'],
                              ropdp.vars['rhoHV [-]'],
                              width=RPARAMS[RSITE]['wu_pdp'],
                              copy=True).astype(np.float64))
        ropdp.vars['PhiDP [deg]'] = uphidp
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                            rsnr.vars,
                                            var2plot='PhiDP [deg]')
            tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                            ropdp.vars,
                                            var2plot='PhiDP [deg]')
        # =============================================================================
        # NME ID and removal
        # =============================================================================
        if rdata.site_name.lower() == 'boxpol':
            pathmfscl = MFS_DIR
            if rdata.scandatetime.year == 2017:
                clfmap = np.loadtxt(
                    CLM_DIR + f'boxpol{rdata.scandatetime.year}b'
                    + '_cluttermap_el0.dat')
            else:
                clfmap = np.loadtxt(
                    CLM_DIR + f'boxpol{rdata.scandatetime.year}'
                    + '_cluttermap_el0.dat')
        elif rdata.site_name.lower() == 'juxpol':
            pathmfscl = None
            clfmap = np.loadtxt(CLM_DIR
                                + 'juxpol2021_cluttermap_el0.dat')
            # clfmap = None
        else:
            pathmfscl = None
            rdata2 = tpx.Rad_scan(scan, RPARAMS[RSITE]['site_name'])
            rdata2.ppi_dwd(get_rvar='cmap')
            clfmap = 1 - tp.utils.radutilities.normalisenanvalues(
                rdata2.vars['cmap [class]'],
                np.nanmin(rdata2.vars['cmap [class]']),
                np.nanmax(rdata2.vars['cmap [class]']))
            clfmap = np.nan_to_num(clfmap, nan=1e-5)

        rnme = tp.eclass.nme.NME_ID(rdata)
        # Despeckle and removal of linear signatures
        rnme.lsinterference_filter(rdata.georef, rdata.params, ropdp.vars,
                                   rhv_min=RPARAMS[RSITE]['rhvmin'],
                                   data2correct=ropdp.vars,
                                   plot_method=PLOT_METHODS)
        # Clutter and anomalous propagation iD and removal
        rnme.clutter_id(rdata.georef, rdata.params, rnme.vars,
                        path_mfs=pathmfscl, min_snr=rsnr.min_snr,
                        binary_class=RPARAMS[RSITE]['bclass'], clmap=clfmap,
                        data2correct=rnme.vars, plot_method=PLOT_METHODS)
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                               rnme.vars)
        # ============================================================================
        # Melting layer allocation
        # ============================================================================
        rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
        if ((abs((mlh_dt - rmlyr.scandatetime).total_seconds())/60)
                <= max_diffdtmin) and mlyrhv:
            rmlyr.ml_bottom = mlyrhv[idx_mlh].ml_bottom
            rmlyr.ml_top = mlyrhv[idx_mlh].ml_top
            rmlyr.ml_thickness = mlyrhv[idx_mlh].ml_thickness
            if (np.isnan(rmlyr.ml_thickness) and ~np.isnan(rmlyr.ml_bottom)):
                rmlyr.ml_thickness = rmlyr.ml_top - rmlyr.ml_bottom
            print(f'{rmlyr.site_name} ML_h in database')
        elif (~np.isnan(mlt_avg) and ~np.isnan(mlk_avg)):
            rmlyr.ml_top = mlt_avg
            rmlyr.ml_thickness = mlk_avg
            rmlyr.ml_bottom = rmlyr.ml_top - rmlyr.ml_thickness
            print(f'{rmlyr.site_name} ML_h in database lies too far,'
                  + ' using the day-average value')
        elif (np.isnan(mlt_avg) and np.isnan(mlk_avg)):
            print(f'{rmlyr.site_name} ML_h not in database,'
                  + ' using the preset value')
            rmlyr.ml_top = RPARAMS[RSITE]['mlt']
            rmlyr.ml_thickness = RPARAMS[RSITE]['mlk']
            rmlyr.ml_bottom = rmlyr.ml_top-rmlyr.ml_thickness
        # PPI MLYR
        rmlyr.ml_ppidelimitation(rdata.georef, rdata.params, rsnr.vars)
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                               rnme.vars, mlyr=rmlyr)
        # =============================================================================
        # ZDR offset correction
        # =============================================================================
        rozdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
        if ((abs((zdr0_dt - rozdr.scandatetime).total_seconds())/60)
                <= max_diffdtmin):
            # print(f'{rozdr.site_name}_ZDR offset in database')
            rozdr.zdr_offset = zdrOv[idx_zdr0].zdr_offset
            if rozdr.zdr_offset == 0 or np.isnan(rozdr.zdr_offset):
                print(f'{rozdr.site_name}_ZDR offset invalid --'
                      + ' using fixed value')
                rozdr.zdr_offset = RPARAMS[RSITE]['zdr_offset']
            else:
                print(f'{rozdr.site_name}_ZDR offset in database')
        else:
            print(f'{rozdr.site_name}_ZDR offset dt in database'
                  + ' far too long -- using fixed value')
            rozdr.zdr_offset = RPARAMS[RSITE]['zdr_offset']
        rozdr.offset_correction(rnme.vars['ZDR [dB]'],
                                zdr_offset=rozdr.zdr_offset,
                                data2correct=rnme.vars)
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                            rnme.vars, var2plot='ZDR [dB]')
            tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                            rozdr.vars,
                                            var2plot='ZDR [dB]')
        # =============================================================================
        # ZH Attenuation Correction
        # =============================================================================
        rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
        if ((abs((pclass_dt - rattc.scandatetime).total_seconds())/60)
                <= max_diffdtmin):
            pcp_its = profs_data['pcp_type'][idx_pclass]
        else:
            pcp_its = 1
        if rband == 'C' and (pcp_its == 0 or pcp_its == 1 or pcp_its == 4):
            att_alpha = [0.05, 0.1, 0.08]  # Light to moderate rain
            rb_a = 0.39  # Continental
            print(f'scan LR-{rband}-{att_alpha}')
        elif rband == 'C' and (pcp_its == 2 or pcp_its == 5):
            att_alpha = [0.1, 0.18, 0.08]  # Moderate to heavy rain
            # rb_a = 0.39  # Continental
            rb_a = 0.27  # Continental
            print(f'scan MR-{rband}-{att_alpha}')
        elif rband == 'C' and (pcp_its == 3 or pcp_its == 6):
            att_alpha = [0.1, 0.18, 0.08]  # Moderate to heavy rain
            rb_a = 0.14  # Tropical
            # att_alpha = [0.05, 0.18, 0.11]  # Light - heavy rain
            print(f'scan HR-{rband}-{att_alpha}')
        if rband == 'X' and (pcp_its == 0 or pcp_its == 1 or pcp_its == 4):
            att_alpha = [0.15, 0.30, 0.28]  # Light rain PARK
            rb_a = 0.19  # Continental
            print(f'scan LR-{rband}-{att_alpha}')
        elif rband == 'X' and (pcp_its == 2 or pcp_its == 5):
            att_alpha = [0.30, 0.45, 0.28]  # Moderate to heavy rain PARK
            rb_a = 0.17  # Continental
            print(f'scan MR-{rband}-{att_alpha}')
        elif rband == 'X' and (pcp_its == 3 or pcp_its == 6):
            att_alpha = [0.30, 0.45, 0.28]  # Moderate to heavy rain PARK
            rb_a = 0.14  # Tropical
            # att_alpha = [0.15, 0.35, 0.22]  # Light - heavy rain
            print(f'scan HR-{rband}-{att_alpha}')
        # Processing of PhiDP for attenuation correction
        rattc.attc_phidp_prepro(
            rdata.georef, rdata.params, rozdr.vars, rhohv_min=0.85,
            phidp0_correction=(True if (rattc.site_name == 'Flechtdorf')
                               or (rattc.site_name == 'Neuheilenbach')
                               or (rattc.site_name == 'Offenthal')
                               else False))
        if ('xpol' in rattc.site_name.lower()
           or rattc.site_name.lower() == 'aachen'):
            rattc.zh_correction(
                rdata.georef, rdata.params, rattc.vars,
                rnme.nme_classif['classif [EC]'], mlyr=rmlyr,
                attc_method='ABRI', pdp_dmin=1, pdp_pxavr_azm=3,
                pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                phidp0=0, coeff_alpha=att_alpha,
                # coeff_a=[9.781e-5, 1.749e-4, 1.367e-4],  # Park
                # coeff_b=[0.757, 0.804, 0.78],  # Park
                coeff_a=[5.50e-5, 1.62e-4, 9.745e-05],  # Diederich
                coeff_b=[0.74, 0.86, 0.8],  # Diederich
                plot_method=PLOT_METHODS)
        else:
            rattc.zh_correction(
                rdata.georef, rdata.params, rattc.vars,
                rnme.nme_classif['classif [EC]'], mlyr=rmlyr,
                attc_method='ABRI', pdp_dmin=1, pdp_pxavr_azm=3,
                pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                phidp0=0, coeff_alpha=att_alpha,
                # coeff_a=[1.59e-5, 4.27e-5, 2.49e-05],  # Diederich
                # coeff_b=[0.73, 0.77, 0.755],  # Diederich
                coeff_a=[1e-5, 4.27e-5, 3e-05],  # MRR+Diederich
                coeff_b=[0.73, 0.85, 0.78],  # MRR+Diederich
                plot_method=PLOT_METHODS)
        # =============================================================================
        # PBBc and ZHAH
        # =============================================================================
        rzhah = tp.attc.r_att_refl.Attn_Refl_Relation(rdata)
        rzhah.ah_zh(rattc.vars, zh_upper_lim=55, temp=temp, rband=rband,
                    copy_ofr=True, data2correct=rattc.vars)
        rattc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']
        mov_avrgf_len = (1, 7)
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
                var2plot1='ZH [dBZ]', var2plot2='ZH+ [dBZ]')
        # =============================================================================
        # ZDR Attenuation Correction
        # =============================================================================
        zhzdr_a = 0.000249173
        zhzdr_b = 2.33327
        if rband == 'C':
            zdr_attc = (9, 5, 3)
        else:
            zdr_attc = (7, 10, 5)
        rattc.zdr_correction(rdata.georef, rdata.params, rozdr.vars,
                             rzhah.vars, rnme.nme_classif['classif [EC]'],
                             mlyr=rmlyr, attc_method='BRI',
                             coeff_beta=RPARAMS[RSITE]['beta'],
                             beta_alpha_ratio=rb_a,
                             rhv_thld=RPARAMS[RSITE]['rhvatt'],
                             mov_avrgf_len=zdr_attc[0],
                             minbins=zdr_attc[1], p2avrf=zdr_attc[2],
                             zh_zdr_model='exp',
                             rparams={'coeff_a': zhzdr_a,
                                      'coeff_b': zhzdr_b},
                             plot_method=PLOT_METHODS)
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_ppidiff(
                rdata.georef, rdata.params, rozdr.vars, rattc.vars,
                var2plot1='ZDR [dB]', var2plot2='ZDR [dB]',
                diff_lims=[-1, 1, .1])
        # =============================================================================
        # KDP Derivation
        # =============================================================================
        # KDP Vulpiani
        if rband == 'C':
            zh_kdp = 'ZH+ [dBZ]'
        elif rband == 'X':
            zh_kdp = 'ZH+ [dBZ]'
        rkdpv = {}
        kdp_vulp = kdpvpi(
            rattc.vars['PhiDP [deg]'], dr=rdata.params['gateres [m]']/1000,
            winlen=RPARAMS[RSITE]['kdpwl'], copy=True)
        rkdpv['PhiDP [deg]'] = kdp_vulp[0]
        rkdpv['KDP [deg/km]'] = kdp_vulp[1]
        # Remove NME
        rattc.vars['KDP* [deg/km]'] = np.where(
            rnme.nme_classif['classif [EC]'] != 0, np.nan,
            rkdpv['KDP [deg/km]'])
        # rattc.vars['KDP* [deg/km]'] = np.where(
        #     rnme.ls_dsp_class['classif [EC]'] != 0, np.nan,
        #     rkdpv['KDP [deg/km]'])
        # Remove negative KDP values in rain region and within ZH threshold
        rattc.vars['KDP* [deg/km]'] = np.where(
            (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
            & (rkdpv['KDP [deg/km]'] < 0) & (rattc.vars[zh_kdp] > 5),
            0, rattc.vars['KDP* [deg/km]'])
        # Patch KDP* using KDP+ using thresholds in ZH and rhoHV
        rattc.vars['KDP+ [deg/km]'] = np.where(
            (rattc.vars[zh_kdp] >= 40) & (rattc.vars[zh_kdp] < 55)
            & (rozdr.vars['rhoHV [-]'] >= 0.95)
            & (~np.isnan(rattc.vars['KDP [deg/km]']))
            & (rattc.vars['KDP [deg/km]'] != 0),
            rattc.vars['KDP [deg/km]'], rattc.vars['KDP* [deg/km]'])
        if PLOT_METHODS:
            tp.datavis.rad_display.plot_ppidiff(
                rdata.georef, rdata.params, rattc.vars, rattc.vars,
                var2plot1='KDP* [deg/km]', var2plot2='KDP+ [deg/km]',
                diff_lims=[-1, 1, 0.1],
                vars_bounds={'KDP [deg/km]': (-1, 3, 17)})
        # =============================================================================
        # Create a new Towerpy object to store processed data
        # =============================================================================
        rd_qcatc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
        rd_qcatc.georef = rdata.georef
        rd_qcatc.params = rdata.params
        rd_qcatc.vars = dict(rattc.vars)
        rd_qcatc.beta_alpha_ratio = rb_a
        rd_qcatc.min_snr = rsnr.min_snr
        rd_qcatc.zh_offset = zh_off
        rd_qcatc.phidp0 = ropdp.phidp_offset
        rd_qcatc.phidp0_qvps = ropdp.phidp_offset_qvps
        rd_qcatc.zdr_offset = rozdr.zdr_offset
        rd_qcatc.ml_top = rmlyr.ml_top
        rd_qcatc.ml_thickness = rmlyr.ml_thickness
        rd_qcatc.ml_bottom = rmlyr.ml_bottom
        rd_qcatc.prf_type = pcp_its
        # rd_qcatc.alpha_ah = np.nanmean(
        #     [np.nanmean(i) for i in rattc.vars['alpha [-]']])
        # del rd_qcatc.vars['alpha [-]']
        # del rd_qcatc.vars['beta [-]']
        del rd_qcatc.vars['ZH* [dBZ]']
        del rd_qcatc.vars['PhiDP* [deg]']
        # del rd_qcatc.vars['PIA [dB]']
        # del rd_qcatc.vars['KDP [deg/km]']
        # del rd_qcatc.vars['PhiDP [deg]']
        # =============================================================================
        # Save file
        # =============================================================================
        fnamedt = (rdata.scandatetime.strftime("%Y%m%d%H%M%S_"))
        print(f'{rozdr.site_name}_ZDR_O '
              + f'[{rozdr.zdr_offset:.2f} dB]')
        print(f'{ropdp.site_name}_PhiDP_O '
              + f'[{ropdp.phidp_offset:.2f} deg]')
        print(f'{rozdr.site_name}_ZH_O '
              + f'[{zh_off:.2f} dBZ]')
        with open(RES_DIR+fnamedt+RPARAMS[RSITE]['site_name']
                  + '_rdqc.tpy', 'wb') as f:
            pickle.dump(rd_qcatc, f, pickle.HIGHEST_PROTOCOL)
        print(f'{fnamedt[:-1]} :: DONE')
    except Exception as err:
        log_file = open(RES_DIR+'log.txt', 'a')
        log_file.write(
            f'{dt.datetime.now().strftime("%Y-%m-%d--%H:%M:%S:")}'
            + f'Error in {rdata.file_name}- {err}'+'\n')
        pass

toc1 = perf_counter()
print(f'TIME ELAPSED [Quality-control]: {dt.timedelta(seconds=toc1-tic)}')
