#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:48:03 2025

@author: dsanchez
"""

from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

rcomp = 'rcomp_qpe_dwd'
# rcomp = 'rcomp_qpe_dwdbxp'
# rcomp = 'rcomp_qpe_dwdjxp'
rcomp = 'rcomp_qpe_dwdxpol'
# rcomp = 'rcomp_qpe_xpol'

WDIR = EWDIR + f'pd_rdres/qpe_all/{rcomp}/rcoeffs/'

SAVE_FIGS = False

RQPE_FILES = [WDIR + i for i in sorted(os.listdir(WDIR))]

RSITES = {'Boxpol': [], 'Juxpol': [], 'Essen': [], 'Flechtdorf': [],
          'Neuheilenbach': [], 'Offenthal': []}

for rs in RQPE_FILES:
    with open(rs, 'rb') as breader:
        rcoeffs1 = pickle.load(breader)
        for k1 in rcoeffs1.keys():
            if k1 in RSITES.keys():
                RSITES[k1].append(rcoeffs1[k1])

rbands = ['C', 'X']

coeffs_rs = {
    k1: {'r_zopt_a':
         np.hstack([rs1['r_zopt']['coeff_a'] for rs1 in rs
                    if (rs1['r_zopt']['coeff_b'] == 1.6 and 'xpol' not in k1)
                    or (rs1['r_zopt']['coeff_a'] != 72 and 'xpol' in k1)
                    ]),
         'r_zopt_b':
         np.hstack([rs1['r_zopt']['coeff_b'] for rs1 in rs
                    if (rs1['r_zopt']['coeff_b'] == 1.6 and 'xpol' not in k1)
                    or (rs1['r_zopt']['coeff_a'] != 72 and 'xpol' in k1)
                    ]),
         'r_kdpopt_a':
         np.hstack([rs1['r_kdpopt']['coeff_a'] for rs1 in rs
                    if (rs1['r_kdpopt']['coeff_b'] > 0.73 and 'xpol' not in k1)
                    or (rs1['r_kdpopt']['coeff_b'] == 0.80 and 'xpol' in k1)
                    ]),
         'r_kdpopt_b':
         np.hstack([rs1['r_kdpopt']['coeff_b'] for rs1 in rs
                    if (rs1['r_kdpopt']['coeff_b'] > 0.73 and 'xpol' not in k1)
                    or (rs1['r_kdpopt']['coeff_b'] == 0.80 and 'xpol' in k1)
                    ])}
    for k1, rs in RSITES.items() if rs}

for k1, v1 in coeffs_rs.items():
    if 'xpol' in k1.lower():
        v1['rband'] = 'X'
    else:
        v1['rband'] = 'C'

bins_zh_x = np.arange(0, 250, 5)
bins_kdp_x = np.arange(0, 30, 1)
bins_zh_c = np.arange(0, 500, 5)
bins_kdp_c = np.arange(0, 30, 1)

coeffs_rs['BoXPol'] = coeffs_rs.pop('Boxpol')
coeffs_rs['JuXPol'] = coeffs_rs.pop('Juxpol')
# %%
# palette = 'crest_r'
# palette = 'Set3'
palettec = plt.colormaps['Dark2'](np.linspace(0.4, 1, 4))
palettex = plt.colormaps['Dark2'](np.linspace(0., 0.6, 4))

# palette = 'icefire'
# palette = sns.diverging_palette(145, 300, s=60)
# palette = sns.light_palette('seagreen')
for rb in rbands:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.despine(fig)
    ax = axs[0]
    sns.histplot(x=np.hstack([v1['r_zopt_a'] for k1, v1 in coeffs_rs.items()
                              if v1['rband'] == rb]),
                 bins=(bins_zh_c if rb == 'C' else bins_zh_x), fill=False,
                 element='step', kde=True, color='k', ax=axs[0])
    sns.histplot({k1: v1['r_zopt_a'] for k1, v1 in coeffs_rs.items()
                  if v1['rband'] == rb},
                 bins=(bins_zh_c if rb == 'C' else bins_zh_x),
                 kde=False, color='k', multiple="stack",
                 palette=(palettex if rb == 'X' else palettec),
                 edgecolor=".3", linewidth=.5, ax=axs[0])
    sns.move_legend(ax, 'upper right', fontsize=12)
    ax.grid(axis='y', which='both')
    ax.set_axisbelow(True)
    ax.set_xlim((0, 350) if rb == 'C' else (0, 160))
    ax.set_xlabel('Coeff a', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('$R(Z_H)[opt] -> Z_H=aR^b [a = opt,'
                 + f' b={1.6 if rb == "C" else 2.14}]$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax = axs[1]
    sns.histplot(x=np.hstack([v1['r_kdpopt_a'] for k1, v1 in coeffs_rs.items()
                              if v1['rband'] == rb]), color='k',
                 bins=(bins_kdp_c if rb == 'C' else bins_kdp_x),
                 ax=ax, element="step", fill=False,
                 kde=(True if rb == 'X' else False))
    sns.histplot({k1: v1['r_kdpopt_a'] for k1, v1 in coeffs_rs.items()
                  if v1['rband'] == rb}, kde=False, color='k',
                 multiple="stack", edgecolor=".3", linewidth=.5, ax=ax,
                 palette=(palettex if rb == 'X' else palettec),
                 bins=(bins_kdp_c if rb == 'C' else bins_kdp_x),)
    ax.grid(axis='y', which='both')
    ax.set_axisbelow(True)
    ax.set_xlim((0, 30))
    ax.set_xlabel('Coeff a', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    sns.move_legend(ax, 'upper right', fontsize=12)
    ax.set_title('$R(K_{DP})[opt] -> R=aK_{DP}^b [a = opt,'
                 + f' b={0.80 if rb == "X" else "opt"}]$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.suptitle(f'{rb}-band radars', fontsize=16)
    fig.tight_layout()
    if SAVE_FIGS:
        len_rdtsets = len(RSITES[k1])
        RES_DIR2 = LWDIR + 'pd_rdres/qpe_all/rcoeffs/'
        if rb == 'C':
            sfx = 'c'
        else:
            sfx = 'x'
        fname = (f"rzhkdpopt{sfx}_{len_rdtsets}_devents.png")
        plt.savefig(RES_DIR2 + fname, dpi=200, format='png')
# %%

# for k1, v1 in coeffs_rs.items():
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     if 'xpol' in k1:
#         axs[0].set_title('$R(Z_H)[opt] -> Z_H=aR^b [a = opt, b=2.14]$')
#         axs[0].hist(v1['r_zopt_a'], bins_zh_x, histtype='step', density=False)
#         # y = norm.pdf(bins_zh_x, np.mean(v1[0]),
#         #              np.nanstd(v1[0], axis=-1, ddof=1))
#         # axs[0].plot(bins_zh_x, y, 'r--')
#     else:
#         axs[0].set_title('$R(Z_H)[opt] -> Z_H=aR^b [a = opt, b=1.6]$')
#         axs[0].hist(v1['r_zopt_a'], bins_zh_c, histtype='step')
#     if 'xpol' in k1:
#         axs[1].hist(v1['r_kdpopt_a'], bins_kdp_x, histtype='step')
#         axs[1].set_title('step')
#         axs[1].set_title('$R(K_{DP})[opt] -> R=aK_{DP}^b [a = opt, b=0.80]$')
#     else:
#         axs[1].hist(v1['r_kdpopt_a'], bins_kdp_c, histtype='step')
#         axs[1].set_title('step')
#         axs[1].set_title('$R(K_{DP})[opt] -> R=aK_{DP}^b [a = opt, b=opt]$')
#     # axs[0, 0].hist(v1[0], bins, density=False, histtype='stepfilled',
#     #                facecolor='g', alpha=0.75)
#     axs[0].set_ylabel('Frequency')
#     axs[0].set_xlabel('Coeff a')
#     axs[1].set_xlabel('Coeff a')
#     fig.suptitle(f'{k1}')
#     fig.tight_layout()
#     plt.show()

# %%

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# rband = 'X'
# for k1, v1 in coeffs_rs.items():
#     if v1['rband'] == rband:
#         axs[0].set_title('$R(Z_H)[opt] -> Z_H=aR^b [a = opt, b=2.14]$')
#         axs[0].hist(np.hstack([v1['r_zopt_a']
#                                for k1, v1 in coeffs_rs.items()
#                                if v1['rband'] == rband]),
#                     bins_zh_x, histtype='stepfilled',  # rwidth=0.8,
#                     density=False, fc='tab:grey')
#         axs[0].hist(v1['r_zopt_a'], bins_zh_x, histtype='step', density=False,
#                     label=k1, ls='-')
#         # y = norm.pdf(
#         #     bins_zh_x, np.nanmean(np.hstack([v1['r_zopt_a']
#         #                                   for k1, v1 in coeffs_rs.items()
#         #                                   if v1['rband'] == 'X'])),
#         #     np.nanstd(np.hstack([v1['r_zopt_a']
#         #                          for k1, v1 in coeffs_rs.items()
#         #                          if v1['rband'] == 'X']), axis=-1, ddof=1))
#         # axs[0].plot(bins_zh_x, y, 'r--')
#         axs[0].set_ylabel('Frequency')
#         axs[0].set_xlabel('Coeff a')
#         # axs[0].legend()
#     if v1['rband'] == rband:
#         axs[1].hist(np.hstack([v1['r_kdpopt_a']
#                                for k1, v1 in coeffs_rs.items()
#                                if v1['rband'] == rband]),
#                     bins_kdp_x, histtype='stepfilled',  # rwidth=0.8,
#                     density=False, fc='tab:grey')
#         axs[1].hist(v1['r_kdpopt_a'], bins_kdp_x, histtype='step',
#                     density=False, label=k1, ls='-')
#         axs[1].set_title('step')
#         axs[1].set_title('$R(K_{DP})[opt] -> R=aK_{DP}^b [a = opt, b=0.80]$')
#         axs[1].legend()
#         axs[1].set_xlabel('Coeff a')
#         # axs[1].grid()
#     fig.suptitle(f'{rband}-band radars')
#     # fig.legend()
#     fig.tight_layout()
#     plt.show()

# rband = 'X'

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
# sns.despine(fig)

# ax = axs[0]
# sns.histplot(x=np.hstack([v1['r_zopt_a'] for k1, v1 in coeffs_rs.items()
#                           if v1['rband'] == rband]), bins=bins_zh_x,
#              kde=True, color='k', edgecolor='w', ax=axs[0])

# # for k1, v1 in coeffs_rs.items():
# #     if v1['rband'] == 'C':
# #         sns.histplot(v1,
# #                      # hue=[k1 for k1, v1 in coeffs_rs.items()
# #                      #           if v1['rband'] == 'C'],
# #                      bins=bins_zh_c, kde=False, element="step", x='r_zopt_a',
# #                      fill=False)

# sns.histplot({k1: v1['r_zopt_a'] for k1, v1 in coeffs_rs.items()
#               if v1['rband'] == rband},
#              bins=bins_zh_x, kde=False, color='k',  multiple="stack",
#              palette="vlag", edgecolor=".3", linewidth=.5, ax=axs[0])
# ax.grid(axis='y', which='both')
# ax.set_axisbelow(True) 
# ax.set_xlim((0, 175))
# ax.set_xlabel('Coeff a')
# ax.set_title('$R(Z_H)[opt] -> Z_H=aR^b [a = opt, b=2.14]$')

# ax = axs[1]
# sns.histplot(x=np.hstack([v1['r_kdpopt_a'] for k1, v1 in coeffs_rs.items()
#                           if v1['rband'] == rband]), bins=bins_kdp_x,
#              kde=True, color='k', edgecolor='w', ax=ax)

# sns.histplot({k1: v1['r_kdpopt_a'] for k1, v1 in coeffs_rs.items()
#               if v1['rband'] == rband},
#              bins=bins_kdp_x, kde=False, color='k',  multiple="stack",
#              palette="vlag", edgecolor=".3", linewidth=.5, ax=ax)
# ax.set_xlim((0, 25))
# ax.set_xlabel('Coeff a')
# ax.set_title('$R(K_{DP})[opt] -> R=aK_{DP}^b [a = opt, b=opt]$')
# ax.grid(axis='y', which='both')
# ax.set_axisbelow(True) 
# fig.suptitle(f'{rband}-band radars')
# fig.tight_layout()
# %%

# hrscl = {'Essen': 'whitesmoke', 'Flechtdorf': 'orchid',
#          'Neuheilenbach': 'lightgreen', 'Offenthal': 'gold'}
# rband = 'C'

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# axs[0].set_title('$R(Z_H)[opt] -> Z_H=aR^b [a = opt, b=1.6]$')
# axs[1].set_title('$R(K_{DP})[opt] -> R=aK_{DP}^b [a = opt, b=opt]$')

# # xpdf = np.hstack([v1['r_zopt_a'] for k1, v1 in coeffs_rs.items()
# #                   if v1['rband'] == rband])
# # pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(-xpdf**2 / 2)

# # axs[0].plot(xpdf, pdf * len(xpdf) * 1)

# for k1, v1 in coeffs_rs.items():
#     if v1['rband'] == rband:
#         # print(k1)
        
#         # axs[0].hist(np.hstack([v1['r_zopt_a']
#         #                        for k1, v1 in coeffs_rs.items()
#         #                        if v1['rband'] == rband]),
#         #             bins_zh_c, histtype='barstacked',  rwidth=1,
#         #             density=False, fc='steelblue', edgecolor='k')
#         axs[0].hist([v1['r_zopt_a'] for k1, v1 in coeffs_rs.items()
#                      if v1['rband'] == rband],
#                     bins_zh_c, histtype='barstacked', rwidth=1,
#                     density=False, edgecolor='k', stacked=True,
#                     label=k1)
        
        
#         # axs[0].hist(v1['r_zopt_a'], bins_zh_c, histtype='step', label=k1,
#         #             color=hrscl.get(k1), ls='-', density=False)
#         axs[0].set_ylabel('Frequency')
#         axs[0].set_xlabel('Coeff a')
#         axs[0].legend()
#     if v1['rband'] == rband:
#         axs[1].hist(np.hstack([v1['r_kdpopt_a']
#                                for k1, v1 in coeffs_rs.items()
#                                if v1['rband'] == rband]),
#                     bins_kdp_c, histtype='stepfilled',  # rwidth=0.8,
#                     density=False, fc='steelblue')
#         axs[1].hist(v1['r_kdpopt_a'], bins_kdp_c, histtype='step', label=k1,
#                     color=hrscl.get(k1))
#     # axs[0, 0].hist(v1[0], bins, density=False, histtype='stepfilled',
#     #                facecolor='g', alpha=0.75)
#         axs[1].set_xlabel('Coeff a')
#         axs[1].legend()
#     fig.suptitle(f'{rband}-band radars')
#     fig.tight_layout()
#     plt.show()
