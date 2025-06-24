"""
Define radar parameters such as rain estimators or radar variables relations.

Created on Mon May  6 15:58:11 2024

@author: dsanchez
"""

# =============================================================================
# Parameters for X-Band
# =============================================================================
# RQPE
rz_ax, rz_bx = 72, 2.14  # Diederich2015
rkdp_ax, rkdp_bx = 16.9, 0.801  # Diederich2015
# rz_ax, rz_bx = (1/0.098)**(1/0.47), 1/0.47  # Chen2021
# rkdp_ax, rkdp_bx = 15.6, 0.83  # Chen2021
# rah_ax, rah_bx = 38, 0.69  # Chen2021
# rz_ax, rz_bx = (1/0.057)**(1/0.57), 1/0.57  # Chen2023
# rkdp_ax, rkdp_bx = 22.9, 0.76  # Chen2023
# rah_ax, rah_bx = 67, 0.78  # Chen2023
rz_hailax, rz_hailbx = (1/0.035)**(1/0.52), 1/0.52  # Chen2023
rzhzdr_ax, rzhzdr_bx, rzhzdr_cx = 0.0039, 1.07, -5.97  # Bringi2001
rkdpzdr_ax, rkdpzdr_bx, rkdpzdr_cx = 28.6, 0.95, -1.37  # Bringi2001
# =============================================================================
# Radar variables relations
# att_alphax = [0.14, 0.41, 0.28]  # Park2005
att_betax = [0.03, 0.06, 0.05]
alphabetar_x = 0.19  # Ryzhkov2014
ahkdp_ax, ahkdp_bx = 0.28, 1
adpkdp_ax, adpkdp_bx = alphabetar_x*ahkdp_ax, 1
# zdrzh_ax, zdrzh_bx = 4.34e-2, 0.393
zdrzh_ax, zdrzh_bx = 2.49173e-4, 2.33327
# zhah_ax, zhah_bx = 9.745e-05, 0.8  # MD15C
rhvtbxp = (0.93, 1.1)
rhvtjxp = (0.9, 1.1)
thr_zwsnwx = 0
thr_zhailx = 55
f_rz_mlx = 0.6
f_rz_spx = 2.8
f_rz_spx = 1.5
rhohv_minx = 0.6
rhvzdrattc_bxp = 0.98
rhvzdrattc_jxp = 0.95
rhohv_kdpx = 0.96

# =============================================================================
# Parameters for C-Band
# =============================================================================
# RQPE
# dflt_selevc = 'ppi_vol_0.5'
dflt_selevc = 'ppi_pcp'
rz_ac, rz_bc = (1/0.052)**(1/0.57), 1/0.57  # Chen2021
rkdp_ac, rkdp_bc = 20.7, 0.72  # Chen2021
# rah_ac, rah_bc = 307, 0.92  # Chen2021
# radp_ac, radp_bc = 452, 0.98  # Chen2021
# rz_ac, rz_bc = (1/0.026)**(1/0.69), 1/0.69  # Chen2023
# rkdp_ac, rkdp_bc = 30.6, 0.71  # Chen2023
# rah_ac, rah_bc = 427, 0.94  # Chen2023
rz_hailac, rz_hailbc = (1/0.022)**(1/0.61), 1/0.61  # Chen2023
rzhzdr_ac, rzhzdr_bc, rzhzdr_cc = 0.0058, 0.91, -2.09  # Bringi2001
rkdpzdr_ac, rkdpzdr_bc, rkdpzdr_cc = 37.9, 0.89, -0.72  # Bringi2001
# rz_ac, rz_bc = 256, 1.42  # DWD
# rz_ac, rz_bc = (1/0.029)**(1/0.67), 1/0.67  # Borowska2011

# =============================================================================
# Radar variables relations
# att_alphac = [0.05, 0.18, 0.1]  # Troemel2014
# att_betac = [0.002, 0.07, 0.04]
# att_betac = [0.015, 0.07, 0.04]
att_betac = [0.008, 0.1, 0.02]
alphabetar_c = 0.39  # Ryzhkov2014
ahkdp_ac, ahkdp_bc = 0.08, 1
adpkdp_ac, adpkdp_bc = alphabetar_c*ahkdp_ac, 1
zdrzh_ac, zdrzh_bc = 2.49173e-4, 2.33327
# zhah_ac, zhah_bc = 2.49e-05, 0.755  # MD 15C
rhvtc = (0.9, 1.1)
thr_zwsnwc = 0
# thr_zwsnwc = 35
thr_zhailc = 55
f_rz_mlc = 0.6
f_rz_spc = 2.8
f_rz_spc = 1.5
rhohv_minc = 0.3
rhvzdrattc_c = 0.98
rhohv_kdpx = 0.95
# Offenthal -> CLID 2021: 143, 2017:207

# =============================================================================
# DEFAULT PARAMS
# =============================================================================
# ZH offset
zhO = {'bxp': {'20170724': 0.5, '20170725': 0.5,
               '20180516': -1.85, '20180923': -1.85, '20181202': -1.85,
               '20190508': -2.75, '20190511': -2.75, '20190720': -2.75,
               '20200614': 0., '20200617': 0.,
               '20210525': 3.5, '20210620': 3.5, '20210629': 3.5,
               '20210713': 3.5, '20210714': 3.5},
       'jxp': {'20170724': 6, '20170725': 6,
               '20180516': 6, '20180923': 6, '20181202': 6,
               '20190508': 6, '20190511': 6, '20190720': 6,
               '20200614': 6, '20200617': 6,
               '20210525': 5, '20210620': 5, '20210629': 5,
               '20210713': 4.5, '20210714': 4.5},
       }
# Phidp offset preset
pdpO = {'bxp': {'20170724': 84, '20170725': 84,
                '20180516': 84, '20180923': 84, '20181202': 84,
                '20190508': 84, '20190511': 84, '20190720': 84,
                '20200614': 84, '20200617': 84,
                '20210525': 84, '20210620': 84, '20210629': 84,
                '20210713': 84, '20210714': 84},
        'jxp': {'20170724': 117, '20170725': 117,
                '20180516': 117, '20180923': 117, '20181202': 117,
                '20190508': 117, '20190511': 117, '20190720': 117,
                '20200614': 117, '20200617': 117,
                '20210525': 117, '20210620': 117, '20210629': 117,
                '20210713': 117, '20210714': 117},
        'ess': {'20170724': 135, '20170725': 135,
                '20180516': 135, '20180923': 135, '20181202': 173,
                '20190508': 173, '20190511': 173, '20190720': 39,
                '20200614': 39, '20200617': 39,
                '20210525': 39, '20210620': 39, '20210629': 39,
                '20210713': 39, '20210714': 39},
        'fle': {'20170724': -9, '20170725': -9,
                '20180516': 40, '20180923': 40, '20181202': 40,
                '20190508': 40, '20190511': 40, '20190720': 40,
                '20200614': -93, '20200617': -93,
                '20210525': -93, '20210620': -93, '20210629': -93,
                '20210713': -93, '20210714': -93},
        'neu': {'20170724': -145, '20170725': -145,
                '20180516': -145, '20180923': -145, '20181202': -145,
                '20190508': -145, '20190511': -145, '20190720': 9,
                '20200614': 9, '20200617': 9,
                '20210525': 9, '20210620': 9, '20210629': 9,
                '20210713': 9, '20210714': 9},
        'off': {'20170724': 105, '20170725': 105,
                '20180516': 94, '20180923': 94, '20181202': 94,
                '20190508': 94, '20190511': 94, '20190720': 102,
                '20200614': 135, '20200617': 135,
                '20210525': 135, '20210620': 135, '20210629': 135,
                '20210713': 135, '20210714': 135},
        }

nlvl = {'bxp': {'20170724': (24, 28, 0.1), '20170725': (24, 28, 0.1),
                '20180516': (25, 29, 0.1), '20180923': (26, 30, 0.1),
                '20181202': (26, 30, 0.1),
                '20190508': (24, 28, 0.1), '20190511': (24, 28, 0.1),
                '20190720': (24, 28, 0.1),
                '20200614': (24, 28, 0.1), '20200617': (24, 28, 0.1),
                '20210525': (24, 28, 0.1), '20210620': (24, 28, 0.1),
                '20210629': (24, 28, 0.1),
                '20210713': (21, 25, 0.1), '20210714': (21, 25, 0.1)},
        'jxp': {'20170724': (29, 33, 0.1), '20170725': (29, 33, 0.1),
                '20180516': (29, 33, 0.1), '20180923': (29, 33, 0.1),
                '20181202': (29, 33, 0.1),
                '20190508': (29, 33, 0.1), '20190511': (29, 33, 0.1),
                '20190720': (29, 33, 0.1),
                '20200614': (29, 33, 0.1), '20200617': (29, 33, 0.1),
                '20210525': (29, 33, 0.1), '20210620': (29, 33, 0.1),
                '20210629': (29, 33, 0.1),
                '20210713': (29, 33, 0.1), '20210714': (29, 33, 0.1)},
        'ess': {'20170724': (36, 42, 0.1), '20170725': (36, 42, 0.1),
                '20180516': (36, 42, 0.1), '20180923': (36, 42, 0.1),
                '20181202': (36, 42, 0.1),
                '20190508': (36, 42, 0.1), '20190511': (36, 42, 0.1),
                '20190720': (36, 42, 0.1),
                '20200614': (36, 42, 0.1), '20200617': (36, 42, 0.1),
                '20210525': (36, 42, 0.1), '20210620': (36, 42, 0.1),
                '20210629': (36, 42, 0.1),
                '20210713': (36, 42, 0.1), '20210714': (36, 42, 0.1)},
        'fle': {'20170724': (36, 42, 0.1), '20170725': (36, 42, 0.1),
                '20180516': (36, 42, 0.1), '20180923': (36, 42, 0.1),
                '20181202': (36, 42, 0.1),
                '20190508': (36, 42, 0.1), '20190511': (36, 42, 0.1),
                '20190720': (36, 42, 0.1),
                '20200614': (36, 42, 0.1), '20200617': (36, 42, 0.1),
                '20210525': (36, 42, 0.1), '20210620': (36, 42, 0.1),
                '20210629': (36, 42, 0.1),
                '20210713': (36, 42, 0.1), '20210714': (36, 42, 0.1)},
        'neu': {'20170724': (36, 42, 0.1), '20170725': (36, 42, 0.1),
                '20180516': (36, 42, 0.1), '20180923': (36, 42, 0.1),
                '20181202': (36, 42, 0.1),
                '20190508': (36, 42, 0.1), '20190511': (36, 42, 0.1),
                '20190720': (36, 42, 0.1),
                '20200614': (36, 42, 0.1), '20200617': (36, 42, 0.1),
                '20210525': (36, 42, 0.1), '20210620': (36, 42, 0.1),
                '20210629': (36, 42, 0.1),
                '20210713': (36, 42, 0.1), '20210714': (36, 42, 0.1)},
        'off': {'20170724': (36, 42, 0.1), '20170725': (36, 42, 0.1),
                '20180516': (36, 42, 0.1), '20180923': (36, 42, 0.1),
                '20181202': (36, 42, 0.1),
                '20190508': (36, 42, 0.1), '20190511': (36, 42, 0.1),
                '20190720': (36, 42, 0.1),
                '20200614': (36, 42, 0.1), '20200617': (36, 42, 0.1),
                '20210525': (36, 42, 0.1), '20210620': (36, 42, 0.1),
                '20210629': (36, 42, 0.1),
                '20210713': (36, 42, 0.1), '20210714': (36, 42, 0.1)},
        'han': {'20170724': (36, 42, 0.1), '20170725': (36, 42, 0.1),
                '20180516': (36, 42, 0.1), '20180923': (36, 42, 0.1),
                '20180923': (36, 42, 0.1),
                '20190508': (36, 42, 0.1), '20190511': (36, 42, 0.1),
                '20190720': (36, 42, 0.1),
                '20200614': (36, 42, 0.1), '20200617': (36, 42, 0.1),
                '20210525': (36, 42, 0.1), '20210620': (36, 42, 0.1),
                '20210629': (36, 42, 0.1),
                '20210713': (36, 42, 0.1), '20210714': (36, 42, 0.1)},
        }

mlyr = {'bxp': 3.0, 'jxp': 3.0, 'ess': 3.0, 'fle': 3.0, 'neu': 3.0,
        'off': 3.0, 'han': 3.0}  # 2018/2019/20210714

# =============================================================================
# Definition dict
# =============================================================================
RPARAMS = [
    {'site_name': 'Boxpol', 'rband': 'X', 'elev': 'n_ppi_010deg',
     'rhvtc': rhvtbxp, 'signvel': -1, 'signpdp': 1, 'bclass': 205,
     'zdr_offset': 0., 'zh_offset': zhO['bxp'], 'phidp_prst': pdpO['bxp'],
     'mlk': 0.8, 'mlt': mlyr['bxp'], 'rhvmin': rhohv_minx, 'nlvl': nlvl['bxp'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betax, 'alphabetar': alphabetar_x,
     'rhvatt': rhvzdrattc_bxp,
     # 'zhah_a': zhah_ax, 'zhah_b': zhah_bx,
     'rz_a': rz_ax, 'rz_b': rz_bx, 'rkdp_a': rkdp_ax, 'rkdp_b': rkdp_bx,
     'rzhzdr_a': rzhzdr_ax, 'rzhzdr_b': rzhzdr_bx, 'rzhzdr_c': rzhzdr_cx,
     'rkdpzdr_a': rkdpzdr_ax, 'rkdpzdr_b': rkdpzdr_bx, 'rkdpzdr_c': rkdpzdr_cx,
     'rz_haila': rz_hailax, 'rz_hailb': rz_hailbx, 'thr_zwsnw': thr_zwsnwx,
     'thr_zhail': thr_zhailx, 'f_rz_ml': f_rz_mlx, 'f_rz_sp': f_rz_spx,
     'ahkdp_a': ahkdp_ax, 'ahkdp_b': ahkdp_bx,
     'zdrzh_a': zdrzh_ax, 'zdrzh_b': zdrzh_bx,
     'adpkdp_a': adpkdp_ax, 'adpkdp_b': adpkdp_bx},
    {'site_name': 'Juxpol', 'rband': 'X', 'elev': 'sweep_9',
     'rhvtc': rhvtjxp, 'signvel': -1, 'signpdp': 1,  'bclass': 143+64,
     'zdr_offset': 1.5, 'zh_offset': zhO['jxp'], 'phidp_prst': pdpO['jxp'],
     'mlk': 0.8, 'mlt': mlyr['jxp'], 'rhvmin': rhohv_minx, 'nlvl': nlvl['jxp'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betax, 'alphabetar': alphabetar_x,
     'rhvatt': rhvzdrattc_jxp,
     # 'zhah_a': zhah_ax, 'zhah_b': zhah_bx,
     'rz_a': rz_ax, 'rz_b': rz_bx, 'rkdp_a': rkdp_ax, 'rkdp_b': rkdp_bx,
     'rzhzdr_a': rzhzdr_ax, 'rzhzdr_b': rzhzdr_bx, 'rzhzdr_c': rzhzdr_cx,
     'rkdpzdr_a': rkdpzdr_ax, 'rkdpzdr_b': rkdpzdr_bx, 'rkdpzdr_c': rkdpzdr_cx,
     'rz_haila': rz_hailax, 'rz_hailb': rz_hailbx, 'thr_zwsnw': thr_zwsnwx,
     'thr_zhail': thr_zhailx, 'f_rz_ml': f_rz_mlx, 'f_rz_sp': f_rz_spx,
     'ahkdp_a': ahkdp_ax, 'ahkdp_b': ahkdp_bx, 'zdrzh_a': zdrzh_ax,
     'zdrzh_b': zdrzh_bx, 'adpkdp_a': adpkdp_ax, 'adpkdp_b': adpkdp_bx},
    {'site_name': 'aaxpol', 'rband': 'X', 'rhvtc': -0, 'bclass': 0,
     'minh90': 1.1, 'elev': 'sweep_4', 'signvel': -1, 'signpdp': -1,
     'zdr_offset': 0, 'clfmap': None, 'phidp_prst': 0, 'kdpwl': 9,
     'rz_a': rz_ax, 'rz_b': rz_bx, 'rkdp_a': rkdp_ax, 'rkdp_b': rkdp_bx,
     'alphabetar': alphabetar_x, 'mlt': 3, 'mlk': 0.8},
    {'site_name': 'Essen', 'rband': 'C', 'elev': dflt_selevc,
     'rhvtc': rhvtc, 'signvel': 1, 'signpdp': 1, 'bclass': 207,
     'zdr_offset': 0, 'phidp_prst': pdpO['ess'], 'zh_offset': 0, 'mlk': 0.8,
     'mlt': mlyr['ess'], 'rhvmin': rhohv_minc, 'nlvl': nlvl['ess'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betac, 'alphabetar': alphabetar_c,
     'rhvatt': rhvzdrattc_c,  # 'zhah_a': zhah_ac, 'zhah_b': zhah_bc,
     'rz_a': rz_ac, 'rz_b': rz_bc, 'rkdp_a': rkdp_ac, 'rkdp_b': rkdp_bc,
     'rzhzdr_a': rzhzdr_ac, 'rzhzdr_b': rzhzdr_bc, 'rzhzdr_c': rzhzdr_cc,
     'rkdpzdr_a': rkdpzdr_ac, 'rkdpzdr_b': rkdpzdr_bc, 'rkdpzdr_c': rkdpzdr_cc,
     'rz_haila': rz_hailac, 'rz_hailb': rz_hailbc, 'thr_zwsnw': thr_zwsnwc,
     'thr_zhail': thr_zhailc, 'f_rz_ml': f_rz_mlc, 'f_rz_sp': f_rz_spc,
     'ahkdp_a': ahkdp_ac, 'ahkdp_b': ahkdp_bc, 'adpkdp_a': adpkdp_ac,
     'adpkdp_b': adpkdp_bc, 'zdrzh_a': zdrzh_ac, 'zdrzh_b': zdrzh_bc},
    {'site_name': 'Flechtdorf', 'rband': 'C', 'elev': dflt_selevc,
     'rhvtc': rhvtc, 'signvel': 1, 'signpdp': 1, 'bclass': 207,
     'zdr_offset': 0, 'phidp_prst': pdpO['fle'], 'zh_offset': 0, 'mlk': 0.8,
     'mlt': mlyr['fle'], 'rhvmin': rhohv_minc, 'nlvl': nlvl['fle'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betac, 'alphabetar': alphabetar_c,
     'rhvatt': rhvzdrattc_c,  # 'zhah_a': zhah_ac, 'zhah_b': zhah_bc,
     'rz_a': rz_ac, 'rz_b': rz_bc, 'rkdp_a': rkdp_ac, 'rkdp_b': rkdp_bc,
     'rzhzdr_a': rzhzdr_ac, 'rzhzdr_b': rzhzdr_bc, 'rzhzdr_c': rzhzdr_cc,
     'rkdpzdr_a': rkdpzdr_ac, 'rkdpzdr_b': rkdpzdr_bc, 'rkdpzdr_c': rkdpzdr_cc,
     'rz_haila': rz_hailac, 'rz_hailb': rz_hailbc, 'thr_zwsnw': thr_zwsnwc,
     'thr_zhail': thr_zhailc, 'f_rz_ml': f_rz_mlc, 'f_rz_sp': f_rz_spc,
     'ahkdp_a': ahkdp_ac, 'ahkdp_b': ahkdp_bc, 'adpkdp_a': adpkdp_ac,
     'adpkdp_b': adpkdp_bc, 'zdrzh_a': zdrzh_ac, 'zdrzh_b': zdrzh_bc},
    {'site_name': 'Neuheilenbach', 'rband': 'C', 'elev': dflt_selevc,
     'rhvtc': rhvtc, 'signvel': 1, 'signpdp': 1, 'bclass': 207,
     'zdr_offset': 0, 'phidp_prst': pdpO['neu'], 'zh_offset': 0, 'mlk': 0.8,
     'mlt': mlyr['neu'], 'rhvmin': rhohv_minc, 'nlvl': nlvl['neu'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betac, 'alphabetar': alphabetar_c,
     'rhvatt': rhvzdrattc_c,  # 'zhah_a': zhah_ac, 'zhah_b': zhah_bc,
     'rz_a': rz_ac, 'rz_b': rz_bc, 'rkdp_a': rkdp_ac, 'rkdp_b': rkdp_bc,
     'rzhzdr_a': rzhzdr_ac, 'rzhzdr_b': rzhzdr_bc, 'rzhzdr_c': rzhzdr_cc,
     'rkdpzdr_a': rkdpzdr_ac, 'rkdpzdr_b': rkdpzdr_bc, 'rkdpzdr_c': rkdpzdr_cc,
     'rz_haila': rz_hailac, 'rz_hailb': rz_hailbc, 'thr_zwsnw': thr_zwsnwc,
     'thr_zhail': thr_zhailc, 'f_rz_ml': f_rz_mlc, 'f_rz_sp': f_rz_spc,
     'ahkdp_a': ahkdp_ac, 'ahkdp_b': ahkdp_bc, 'adpkdp_a': adpkdp_ac,
     'adpkdp_b': adpkdp_bc, 'zdrzh_a': zdrzh_ac, 'zdrzh_b': zdrzh_bc},
    {'site_name': 'Offenthal', 'rband': 'C', 'elev': dflt_selevc,
     'rhvtc': rhvtc, 'signvel': 1, 'signpdp': 1, 'bclass': 207,
     'zdr_offset': 0, 'phidp_prst': pdpO['off'], 'zh_offset': 0, 'mlk': 0.8,
     'mlt': mlyr['off'], 'rhvmin': rhohv_minc, 'nlvl': nlvl['off'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betac, 'alphabetar': alphabetar_c,
     'rhvatt': rhvzdrattc_c,  # 'zhah_a': zhah_ac, 'zhah_b': zhah_bc,
     'rz_a': rz_ac, 'rz_b': rz_bc, 'rkdp_a': rkdp_ac, 'rkdp_b': rkdp_bc,
     'rzhzdr_a': rzhzdr_ac, 'rzhzdr_b': rzhzdr_bc, 'rzhzdr_c': rzhzdr_cc,
     'rkdpzdr_a': rkdpzdr_ac, 'rkdpzdr_b': rkdpzdr_bc, 'rkdpzdr_c': rkdpzdr_cc,
     'rz_haila': rz_hailac, 'rz_hailb': rz_hailbc, 'thr_zwsnw': thr_zwsnwc,
     'thr_zhail': thr_zhailc, 'f_rz_ml': f_rz_mlc, 'f_rz_sp': f_rz_spc,
     'ahkdp_a': ahkdp_ac, 'ahkdp_b': ahkdp_bc, 'adpkdp_a': adpkdp_ac,
     'adpkdp_b': adpkdp_bc, 'zdrzh_a': zdrzh_ac, 'zdrzh_b': zdrzh_bc},
    {'site_name': 'Hannover', 'rband': 'C', 'elev': dflt_selevc,
     'rhvtc': rhvtc, 'signvel': 1, 'signpdp': 1, 'bclass': 207,
     'zdr_offset': 0, 'phidp_prst': 0, 'zh_offset': 0, 'mlk': 0.8,
     'mlt': mlyr['han'], 'rhvmin': rhohv_minc, 'nlvl': nlvl['han'],
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betac, 'alphabetar': alphabetar_c,
     'rhvatt': rhvzdrattc_c,  # 'zhah_a': zhah_ac, 'zhah_b': zhah_bc,
     'rz_a': rz_ac, 'rz_b': rz_bc, 'rkdp_a': rkdp_ac, 'rkdp_b': rkdp_bc,
     'rzhzdr_a': rzhzdr_ac, 'rzhzdr_b': rzhzdr_bc, 'rzhzdr_c': rzhzdr_cc,
     'rkdpzdr_a': rkdpzdr_ac, 'rkdpzdr_b': rkdpzdr_bc, 'rkdpzdr_c': rkdpzdr_cc,
     'rz_haila': rz_hailac, 'rz_hailb': rz_hailbc, 'thr_zwsnw': thr_zwsnwc,
     'thr_zhail': thr_zhailc, 'f_rz_ml': f_rz_mlc, 'f_rz_sp': f_rz_spc,
     'ahkdp_a': ahkdp_ac, 'ahkdp_b': ahkdp_bc, 'adpkdp_a': adpkdp_ac,
     'adpkdp_b': adpkdp_bc, 'zdrzh_a': zdrzh_ac, 'zdrzh_b': zdrzh_bc},
    ]

RPRODSLTX = {'r_adp': '$R(A_{DP})$', 'r_ah': '$R(A_{H})$',
             'r_kdp': '$R(K_{DP})$', 'r_z': '$R(Z_H)$',
             'r_ah_kdp': '$R(A_{H}, K_{DP})$',
             'r_kdp_zdr': '$R(K_{DP}, Z_{DR})$', 'r_z_ah': '$R(Z_H, A_{H})$',
             'r_z_kdp': '$R(Z_{H}, K_{DP})$', 'r_z_zdr': '$R(Z_{H}, Z_{DR})$',
             'r_kdpopt': '$R(K_{DP})[opt]$', 'r_zopt': '$R(Z_{H})[opt]$',
             'r_ah_kdpopt': '$R(A_{H}, K_{DP}[opt])$',
             'r_zopt_ah': '$R(Z_{H}[opt], A_{H})$',
             'r_zopt_kdp': '$R(Z_{H}[opt], K_{DP})$',
             'r_zopt_kdpopt': '$R(Z_{H}[opt], K_{DP}[opt])$',
             'r_aho_kdpo': '$R(A_{H}, K_{DP})[evnt-spcf]$',
             'r_kdpo': '$R(K_{DP})[evnt-spcf]$',
             'r_zo': '$R(Z_{H})[evnt-spcf]$',
             'r_zo_ah': '$R(Z_{H}, A_{H})[evnt-spcf]$',
             'r_zo_kdp': '$R(Z_{H}, K_{DP})[evnt-spcf]$',
             'r_zo_zdr': '$R(Z_{H}, Z_{DR})[evnt-spcf]$'}
