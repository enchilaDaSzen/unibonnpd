#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:45:57 2022

@author: dsanchez
"""

import datetime as dt
import numpy as np
import towerpy as tp
import wradlib as wrl
from tqdm import tqdm
# import copy
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from towerpy.utils.radutilities import find_nearest
# from towerpy.base import TowerpyError
from radar import twpext as tpx

# =============================================================================
# Define working directory and list files
# =============================================================================
START_TIME = dt.datetime(2021, 7, 14, 6, 16)
START_TIME = dt.datetime(2021, 7, 14, 12, 50)
# START_TIME = dt.datetime(2017, 7, 24, 5, 16)
STOP_TIME = START_TIME+dt.timedelta(minutes=5)

data4calib = 'qvps'
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/enchiladasz/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'
# PDIR = '/media/enchiladaszen/Samsung1TB/safe/radar_datasets/dwd_xpol/'
# PROFSDATA = EWDIR + 'pd_rdres/20210714/'
MFS_DIR = (LWDIR + 'codes/github/unibonnpd/qc/xpol_mfs/')
CLM_DIR = (LWDIR + 'codes/github/unibonnpd/qc/xpol_clm/')

# RADAR_SITE = 'JuxPol'
# scans_elevs = [28., 18., 14., 11., 8.2, 6., 4.5, 3.1, 1.7, 0.6]
# lpfiles = tpx.get_listfilesxpol(RADAR_SITE, scans_elevs[0],
#                                       start_time, stop_time,
#                                       # parent_dir=wdir
#                                       )*len(scans_elevs)
RADAR_SITE = 'Boxpol'
scans_elevs = ['n_ppi_280deg', 'n_ppi_180deg', 'n_ppi_140deg', 'n_ppi_110deg',
               'n_ppi_082deg', 'n_ppi_060deg', 'n_ppi_045deg', 'n_ppi_031deg',
               'n_ppi_020deg', 'n_ppi_010deg'
                ]
lpfiles = [tpx.get_listfilesxpol(RADAR_SITE, i, START_TIME, STOP_TIME,
                                 # parent_dir=wdir
                                 ) for i in scans_elevs]

rparams = [
    {'site_name': 'Boxpol', 'minsnr': -23, 'bclass': 159, 'minh90': 1.1,
     'elev': 'n_ppi_010deg', 'signvel': -1, 'signpdp': 1, 'zdr_offset': -.75,
     'phidp_offset': 0, 'zh_offset': 2.75,
     'clfmap': np.loadtxt(CLM_DIR + 'boxpol_cluttermap_el0.dat')},
    {'site_name': 'JuxPol', 'minsnr': 0, 'bclass': 159, 'minh90': 1.1,
      'elev': 0.6, 'signvel': -1, 'signpdp': -1, 'zdr_offset': 0,
      'phidp_offset': 0,
      'clfmap': np.loadtxt(CLM_DIR + 'juxpol_cluttermap_el0.dat')}
    ]

# %%
# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
rdata = [tpx.Rad_scan(i[0], f'{RADAR_SITE}') for i in lpfiles]
[robj.ppi_xpol()
 for i, robj in enumerate(rdata)]
for i, v in enumerate(rdata):
    # v.vars['PhiDP [deg]'] *= -1
    v.vars['ZH [dBZ]'] += next(item['zh_offset'] for item in rparams
                                if item['site_name'] == v.site_name)

# rdata.vars['V [m/s]'] *= -1
# rdata.georefUTM()

#%%
# =============================================================================
# rhoHV calibration
# =============================================================================
lwsei = -1
rcrho = [tpx.rhoHV_Noise_Bias(rd) for rd in rdata]
rcrho[lwsei].iterate_radcst(rdata[lwsei].georef, rdata[lwsei].params,
                            rdata[lwsei].vars, rad_cst=[23, 25, .1],
                            data2correct=rdata[lwsei].vars,
                            # plot_method=True
                             )
# for c, v in enumerate(rdata):
#     v.params['radar constant [dB]'] = rcrho[lwsei].rhohv_corrs['radar constant [dB]']
[v.rhohv_noise_correction(rdata[i].georef, rdata[i].params, rdata[i].vars,
                          rad_cst=rcrho[lwsei].rhohv_corrs['Noise level [dB]'],
                          data2correct=rdata[i].vars)
 for i, v in enumerate(rcrho)]

#%%
# =============================================================================
# PhiDP unfolding
# =============================================================================
for cnt, robj in enumerate(rcrho):
    robj.vars['PhiDP [deg]'] = wrl.dp.unfold_phi(
        rdata[cnt].vars['PhiDP [deg]'], robj.vars['rhoHV [-]'],
        width=5, copy=True)

#%%
# Noise and clutter suppression
# =============================================================================
rsnr = [tp.eclass.snr.SNR_Classif(robj) for robj in rdata]
[v.signalnoiseratio(rdata[i].georef, rdata[i].params, rcrho[i].vars,
                    min_snr=next(item['minsnr'] for item in rparams
                                 if item['site_name'] == v.site_name),
                    data2correct=rcrho[i].vars, plot_method=False
                    )
 for i, v in enumerate(tqdm(rsnr, desc='rsnr_towerpy'))]

#%%
rnme = [tp.eclass.nme.NME_ID(rd) for rd in rsnr]
[v.clutter_id(rdata[i].georef, rdata[i].params, rsnr[i].vars,
              binary_class=207-64, path_mfs=MFS_DIR, data2correct=rsnr[i].vars,
              # clmap=next(item['clfmap'] for item in rparams
              #            if item['site_name'] == v.site_name)
              plot_method=False
              )
  for i, v in enumerate(tqdm(rnme, desc='rnmix_towerpy'))
  # if v.elev_angle <= 8
  ]
# for c, i in enumerate(rnme):
#     if i.elev_angle >= 1:
#         i.vars = rsnr[c].vars

#%%
qvpsr = 25
# rscans_georef = [i.georef for i in rdata]
# rscans_params = [i.params for i in rdata]
# rscans_vars = [i.vars for i in rnme]

rdqvps = tp.profs.polprofs.PolarimetricProfiles(rdata[0])
rdqvps.pol_rdqvps([i.georef for i in rdata], [i.params for i in rdata],
                  [i.vars for i in rsnr], spec_range=qvpsr)

#%%
tp.datavis.rad_display.plot_rdqvps([i.georef for i in rdata],
                                   [i.params for i in rdata], rdqvps,
                                   mlyr=None,
                                   ucmap='Spectral', spec_range=qvpsr,
                                   # vars_bounds={'PhiDP [deg]': [80, 93]},
                                   ylims=[0.2, 3], all_desc=True, fig_size=None)

#%%

# =============================================================================
# Interrdqvpspolated QVPS
# =============================================================================

# cm = plt.cm.get_cmap('OrRd_r')(np.linspace(0., .8, len(rdqvps.qvps_itp)))
# cm = plt.cm.get_cmap('tpylsc_pvars_r')(np.linspace(0., .8,
#                                                    len(rdqvps.qvps_itp)))

# fontsizelabels = 20
# fontsizetitle = 25
# fontsizetick = 18
# lpv = {'ZH [dBZ]': [-10, 60], 'ZDR [dB]': [-2, 6],
#        'PhiDP [deg]': [0, 90], 'KDP [deg/km]': [-2, 6],
#        'rhoHV [-]': [0.6, 1], 'LDR [dB]': [-35, 0],
#        'V [m/s]': [-5, 5], 'gradV [dV/dh]': [-1, 0],
#        }
# # r1 = 0
# # r2 = len(beam_height)
# ttxt = f"{rscans_params[0]['datetime']:%Y-%m-%d %H:%M:%S}"

# fig, ax = plt.subplots(1, len(qvpvar), sharey=True)
# fig.suptitle(f'RD-Quasi-Vertical profiles of polarimetric variables' '\n' f'{ttxt}',
#              fontsize=fontsizetitle)
# for c, i in enumerate(qvps_itp):
#     for n, (a, (key, value)) in enumerate(zip(ax.flatten(), i.items())):
#         a.plot(value, yaxis,
#                label=f"{rscans_params[c]['elev_ang [deg]']}" r"$^{\circ}$",
#                alpha=.5,
#                # color=cm[c]
#                )
#         a.set_xlabel(f'{key}', fontsize=fontsizelabels)
#         # if stats:
#         #     a.fill_betweenx(yaxis,
#         #                     value[r1:r2] + stats.get(key, value*np.nan)[r1:r2],
#         #                     value[r1:r2] - stats.get(key, value*np.nan)[r1:r2],
#         #                     alpha=0.4, label='std')
#         if n == 0:
#             a.set_ylabel('Height [km]', fontsize=fontsizelabels, labelpad=15)
#         a.tick_params(axis='both', labelsize=fontsizetick)
#         a.grid(True)
#         a.legend(loc='upper right')
# for n, (a, (key, value)) in enumerate(zip(ax.flatten(), i.items())):
#     a.plot(rdqvps[key], yaxis,
#            'k', lw=3,
#            label='RD-QVP'
#            )
#     a.legend(loc='upper right')
# a.set_ylim(0, 10)


# fig, ax = plt.subplots(1, len(rdqvps.rd_qvps), sharey=True)
# fig.suptitle(f'RD-Quasi-Vertical profiles of polarimetric variables' '\n' f'{ttxt}',
#              fontsize=fontsizetitle)
# for c, i in enumerate(rdqvps.qvps_itp):
#     for n, (a, (key, value)) in enumerate(zip(ax.flatten(), i.items())):
#         a.plot(value, rdqvps.georef['profiles_height [km]'],
#                label=f"{rscans_params[c]['elev_ang [deg]']}" r"$^{\circ}$",
#                # alpha=.5,
#                color=cm[c], ls='--'
#                )
#         a.set_xlabel(f'{key}', fontsize=fontsizelabels)
#         # if stats:
#         #     a.fill_betweenx(yaxis,
#         #                     value[r1:r2] + stats.get(key, value*np.nan)[r1:r2],
#         #                     value[r1:r2] - stats.get(key, value*np.nan)[r1:r2],
#         #                     alpha=0.4, label='std')
#         if n == 0:
#             a.set_ylabel('Height [km]', fontsize=fontsizelabels, labelpad=15)
#         a.tick_params(axis='both', labelsize=fontsizetick)
#         a.grid(True)
#         a.legend(loc='upper right')
# for n, (a, (key, value)) in enumerate(zip(ax.flatten(), i.items())):
#     a.plot(rdqvps.rd_qvps[key], rdqvps.georef['profiles_height [km]'],
#            'k', lw=3,
#            label='RD-QVP'
#            )
#     a.legend(loc='upper right')
# a.set_ylim(0, 10);


#%%

# # pv = 'rhoHV [-]'
# pv = 'ZH [dBZ]'
# # pv = 'ZDR [dB]'


# fig, ax = plt.subplots()

# for c, i in enumerate(rscans_georef):
#     ax.plot(i['range [m]']/1000, i['beam_height [km]'][0], color=cm[c],
#             label=f"{rscans_params[c]['elev_ang [deg]']}" r"$^{\circ}$",)
#     # ax.plot(-i, qvps_h[c], color=cm[c])

# # ax.axvline(spec_range)
# # ax.axvline(-spec_range)
# ax.legend(loc='upper right')

# n = 0
# v2p = 'ZH [dBZ]'
# # v2p = 'ZDR [dB]'
# # v2p = 'PhiDP [deg]'
# v2p = 'rhoHV [-]'
# # v2p = 'V [m/s]'

# # tp.datavis.rad_display.plot_ppi(rdata[n].georef, rdata[n].params,
# #                                        rdata[n].vars,
# #                                        # rsnr[n].vars,
# #                                        var2plot=v2p,
# #                                        # cpy_feats={'status': True},
# #                                        # data_proj=ccrs.UTM(zone=32),
# #                                        # xlims=[4.3, 9.2], ylims=[52.75, 48.75],
# #                                        # xlims=[4.3, 16.5], ylims=[55.5, 46.5]  # Germany
# #                                        # ucmap=cm[3],
# #                                        )
# # for i in qvps_itp:
# #     ax.plot(i[pv], yaxis)
# ax.set_xlim(0., 150)
# ax.set_ylim(0, 18)

# %%

# def plot_rdqvps(tp_rdqvp, mlyr=None, vars_bounds=None, ylims=None,
#                 ucmap=None, spec_range=None):
#     """
#     Display a set of RD-QVPS of polarimetric variables.

#     Parameters
#     ----------
#     tp_rdqvp : PolarimetricProfiles Class
#         Outputs of the RD-QVPs function.
#     mlyr : MeltingLayer Class, optional
#         Plots the melting layer within the polarimetric profiles.
#         The default is None.
#     vars_bounds : dict containing key and 2-element tuple or list, optional
#         Boundaries [min, max] between which radar variables are
#         to be plotted. The default are:
#             {'ZH [dBZ]': [-10, 60],
#              'ZDR [dB]': [-2, 6],
#              'PhiDP [deg]': [0, 180], 'KDP [deg/km]': [-2, 6],
#              'rhoHV [-]': [0.6, 1],
#              'LDR [dB]': [-35, 0],
#              'V [m/s]': [-5, 5], 'gradV [dV/dh]': [-1, 0]}
#     ylims : 2-element tuple or list, optional
#         Set the y-axis view limits [min, max]. The default is None.
#     ucmap : colormap, optional
#         User-defined colormap.
#     spec_range : int, optional
#         Range from the radar within which the data was used.
#     """
#     tpcm = 'tpylsc_pvars_r'
#     cmaph = mpl.colormaps[tpcm](np.linspace(0., .8,
#                                             len(tp_rdqvp.qvps_itp)))
#     if ucmap is not None:
#         cmaph = mpl.colormaps[ucmap](np.linspace(0, 1,
#                                                  len(tp_rdqvp.qvps_itp)))

#     fontsizelabels = 20
#     fontsizetitle = 25
#     fontsizetick = 18
#     lpv = {'ZH [dBZ]': [-10, 60], 'ZDR [dB]': [-2, 6],
#            'PhiDP [deg]': [0, 90], 'KDP [deg/km]': [-2, 6],
#            'rhoHV [-]': [0.6, 1], 'LDR [dB]': [-35, 0],
#            'V [m/s]': [-5, 5], 'gradV [dV/dh]': [-1, 0],
#            }
#     if vars_bounds:
#         lpv.update(vars_bounds)

#     ttxt = f"{rscans_params[0]['datetime']:%Y-%m-%d %H:%M:%S}"

#     fig = plt.figure(layout="constrained")
#     fig.suptitle('RD-Quasi-Vertical profiles of polarimetric variables \n'
#                  f'{ttxt}', fontsize=fontsizetitle)

#     axd = fig.subplot_mosaic("""
#                              ABCD
#                              EEEE
#                              """, sharey=True, height_ratios=[5, 1])

#     for c, i in enumerate(tp_rdqvp.qvps_itp):
#         for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
#             axd[a].plot(value, tp_rdqvp.georef['profiles_height [km]'],
#                         label=(f"{rscans_params[c]['elev_ang [deg]']}"
#                                + r"$^{\circ}$"), color=cmaph[c], ls='--')
#             axd[a].set_xlabel(f'{key}', fontsize=fontsizelabels)
#             if n == 0:
#                 axd[a].set_ylabel('Height [km]', fontsize=fontsizelabels,
#                                   labelpad=10)
#             axd[a].tick_params(axis='both', labelsize=fontsizetick)
#             axd[a].grid(True)
#             axd[a].legend(loc='upper right')
#     for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
#         axd[a].plot(tp_rdqvp.rd_qvps[key],
#                     tp_rdqvp.georef['profiles_height [km]'], 'k', lw=3,
#                     label='RD-QVP')
#         axd[a].legend(loc='upper right')
#         if vars_bounds:
#             if key in lpv:
#                 axd[a].set_xlim(lpv.get(key))
#             else:
#                 axd[a].set_xlim([np.nanmin(value), np.nanmax(value)])
#         if mlyr:
#             axd[a].axhline(mlyr.ml_top, c='tab:blue', ls='dashed', lw=5,
#                            alpha=.5, label='$ML_{top}$')
#             axd[a].axhline(mlyr.ml_bottom, c='tab:purple', ls='dashed', lw=5,
#                            alpha=.5, label='$ML_{bottom}$')
#         if ylims:
#             axd[a].set_ylim(ylims)

#     scan_st = axd['E']
#     for c, i in enumerate(rscans_georef):
#         scan_st.plot(i['range [m]']/1000, i['beam_height [km]'][0],
#                      color=cmaph[c], ls='--',
#                      label=(f"{rscans_params[c]['elev_ang [deg]']}"
#                             + r"$^{\circ}$"))
#         scan_st.plot(i['range [m]']/-1000, i['beam_height [km]'][0],
#                      color=cmaph[c], ls='--')

#     scan_st.tick_params(axis='both', labelsize=fontsizetick-5)
#     if spec_range:
#         scan_st.axvline(spec_range, c='k', lw=3)
#         scan_st.axvline(-spec_range, c='k', lw=3)

# plot_rdqvps(rdqvps, 
#             # ylims=(0, 8),
#             spec_range=50,
#             # vars_bounds={'PhiDP [deg]': (85, 95)}
#             # ucmap='brg'
            
#             )