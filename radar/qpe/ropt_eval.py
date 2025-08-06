#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:48:03 2025

@author: dsanchez
"""

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

LWDIR = '/home/dsanchez/sciebo_dsr/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
WDIR = LWDIR + 'pd_rdres/qpe_all/rsite_qpe_ropt/'

SAVE_FIGS = False

RQPE_FILES = [WDIR + i for i in sorted(os.listdir(WDIR))
              # if i.endswith('rhqpe.tpy')
              # if 'rhqpe' in i
              ]

RSITES = {'Boxpol': [], 'Juxpol': [], 'Essen': [], 'Flechtdorf': [],
          'Neuheilenbach': [], 'Offenthal': []}

# RSITES = {'Boxpol': [], 'Juxpol': []}

for rs in RQPE_FILES:
    if 'box' in rs:
        with open(rs, 'rb') as breader:
            RSITES['Boxpol'].append(pickle.load(breader))
    if 'jux' in rs:
        with open(rs, 'rb') as breader:
            RSITES['Juxpol'].append(pickle.load(breader))
    if 'ess' in rs:
        with open(rs, 'rb') as breader:
            RSITES['Essen'].append(pickle.load(breader))
    if 'fle' in rs:
        with open(rs, 'rb') as breader:
            RSITES['Flechtdorf'].append(pickle.load(breader))
    if 'neu' in rs:
        with open(rs, 'rb') as breader:
            RSITES['Neuheilenbach'].append(pickle.load(breader))
    if 'off' in rs:
        with open(rs, 'rb') as breader:
            RSITES['Offenthal'].append(pickle.load(breader))

coeffs_rs = {k1:
             [np.hstack([[c1[0] for c1 in rz1['rz_opt']
                          if (c1[1] == 1.6 and 'xpol' not in k1)
                          or (c1[0] != 72 and 'xpol' in k1)
                          ] for rz1 in rs]),
              np.hstack([[c1[1] for c1 in rz1['rz_opt']
                          if (c1[1] == 1.6 and 'xpol' not in k1)
                          or (c1[0] != 72 and 'xpol' in k1)
                          ] for rz1 in rs]),
              np.hstack([[c1[0] for c1 in rz1['rkdp_opt']
                          if (c1[1] > 0.73 and 'xpol' not in k1)
                          or (c1[1] == 0.80 and 'xpol' in k1)
                          # or (c1[1] == 0.80 and 'xpol' in k1)
                          ] for rz1 in rs]),
              np.hstack([[c1[1] for c1 in rz1['rkdp_opt']
                          if (c1[1] > 0.73 and 'xpol' not in k1)
                          or (c1[1] == 0.80 and 'xpol' in k1)
                          ] for rz1 in rs])]
             for k1, rs in RSITES.items()}

bins_zh_x = np.arange(0, 150, 1)
bins_kdp_x = np.arange(0, 35, 1)
bins_zh_c = np.arange(0, 350, 5)
bins_kdp_c = np.arange(0, 35, 1)


for k1, v1 in coeffs_rs.items():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    if 'xpol' in k1:
        axs[0].set_title('$R(Z_H)(opt) -> Z_H=aR^b [a = opt, b=2.14]$')
        axs[0].hist(v1[0], bins_zh_x, histtype='step', density=False)
        # y = norm.pdf(bins_zh_x, np.mean(v1[0]),
        #              np.nanstd(v1[0], axis=-1, ddof=1))
        # axs[0].plot(bins_zh_x, y, 'r--')
    else:
        axs[0].set_title('$R(Z_H)(opt) -> Z_H=aR^b [a = opt, b=1.6]$')
        axs[0].hist(v1[0], bins_zh_c, histtype='step')
    
    
    if 'xpol' in k1:
        axs[1].hist(v1[2], bins_kdp_x, histtype='step')
        axs[1].set_title('step')
        axs[1].set_title('$R(K_{DP})(opt) -> R=aK_{DP}^b [a = opt, b=0.80]$')
    else:
        axs[1].hist(v1[2], bins_kdp_c, histtype='step')
        axs[1].set_title('step')
        axs[1].set_title('$R(K_{DP})(opt) -> R=aK_{DP}^b [a = opt, b=opt]$')
    # axs[0, 0].hist(v1[0], bins, density=False, histtype='stepfilled',
    #                facecolor='g', alpha=0.75)
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlabel('Coeff a')
    axs[1].set_xlabel('Coeff a')
    fig.suptitle(f'{k1}')
    fig.tight_layout()
    plt.show()
    
    if SAVE_FIGS:
        len_rdtsets = len(RSITES[k1])
        RES_DIR2 = WDIR + 'resultsh/'
        fname =  (f"{k1.lower()}_{len_rdtsets}devents_rzhkdpopt.png")
        plt.savefig(RES_DIR2 + fname, format='png')

