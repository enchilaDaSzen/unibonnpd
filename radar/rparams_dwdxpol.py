"""
Define radar parameters such as rain estimators or radar variables relations.

Created on Mon May  6 15:58:11 2024

@author: dsanchez
"""

# =============================================================================
# PARAMETERS FOR X-BAND
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
# RADAR VARIABLES RELATIONS AND QC
# att_alphax = [0.14, 0.41, 0.28]  # Park2005
att_betax = [0.03, 0.06, 0.05]
alphabetar_x = 0.19  # Ryzhkov2014
ahkdp_ax, ahkdp_bx = 0.28, 1
adpkdp_ax, adpkdp_bx = alphabetar_x*ahkdp_ax, 1
# zdrzh_ax, zdrzh_bx = 4.34e-2, 0.393
zdrzh_ax, zdrzh_bx = 2.49173e-4, 2.33327
# zhah_ax, zhah_bx = 9.745e-05, 0.8  # MD15C
rhvtbxp = (0.93, 1.1)  # Expected theoretical values of RhoHV in rain, BoXPol
rhvtjxp = (0.90, 1.1)  # Expected theoretical values of RhoHV in rain, JuXPol
rhvtaxp = (0.90, 1.1)  # Expected theoretical values of RhoHV in rain, AaXPol
thr_zwsnwx = 0  # ZH threshold for wet snow, Xband
thr_zhailx = 55  # ZH threshold for hail, Xband
f_rz_mlx = 0.6  # Factor applied to the ZR relation within the melting layer
f_rz_spx = 2.8  # Factor applied to the ZR relation above the melting layer
# f_rz_spx = 1.5
rhohv_minx = 0.6  # Min value of RhohV used to remove interference and LS
rhvzdrattc_bxp = 0.98  # RhohV threshold for rain, BoXPol
rhvzdrattc_jxp = 0.95  # RhohV threshold for rain, JuXPol

# =============================================================================
# PARAMETERS FOR C-BAND
# =============================================================================
dwd_sites = {'ASR Borkum': 'asb', 'Boostedt': 'boo', 'Dresden': 'drs',
             'Eisberg': 'eis', 'Essen': 'ess', 'Feldberg': 'fbg',
             'Flechtdorf': 'fld', 'Hannover': 'hnr', 'Isen': 'isn',
             'Memmingen': 'mem', 'Neuhaus': 'neu', 'Neuheilenbach': 'nhb',
             'Offenthal': 'oft', 'Protzel': 'pro', 'Rostock': 'ros',
             'Turkheim': 'tur', 'Ummendorf': 'umd'}
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
# RADAR VARIABLES RELATIONS AND QC
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
# Offenthal -> CLID 2021: 143, 2017:207

# =============================================================================
# DEFAULT PARAMS
# =============================================================================
# Working dates
workdt = ['20170724', '20170725', '20180516', '20180923', '20181202',
          '20190508', '20190511', '20190720', '20200614', '20200617',
          '20210525', '20210620', '20210629', '20210713', '20210714']
# =============================================================================
# ZH OFFSET
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
       'axp': {'20230714': 0},
       }
zhOdmmy = {dwdrs: {wdt: 0 for wdt in workdt} for dwdrs in dwd_sites.values()
           if dwdrs not in zhO}
zhO = zhO | zhOdmmy
# =============================================================================
# PHIDP OFFSET PRESET
pdpO = {'bxp': {wdt: 84 for wdt in workdt},
        'jxp': {wdt: 117 for wdt in workdt},
        'axp': {wdt: 0 for wdt in workdt},
        'ess': {'20170724': 135, '20170725': 135,
                '20180516': 135, '20180923': 135, '20181202': 173,
                '20190508': 173, '20190511': 173, '20190720': 39,
                '20200614': 39, '20200617': 39,
                '20210525': 39, '20210620': 39, '20210629': 39,
                '20210713': 39, '20210714': 39},
        'fld': {'20170724': -9, '20170725': -9,
                '20180516': 40, '20180923': 40, '20181202': 40,
                '20190508': 40, '20190511': 40, '20190720': 40,
                '20200614': -93, '20200617': -93,
                '20210525': -93, '20210620': -93, '20210629': -93,
                '20210713': -93, '20210714': -93},
        'nhb': {'20170724': -145, '20170725': -145,
                '20180516': -145, '20180923': -145, '20181202': -145,
                '20190508': -145, '20190511': -145, '20190720': 9,
                '20200614': 9, '20200617': 9,
                '20210525': 9, '20210620': 9, '20210629': 9,
                '20210713': 9, '20210714': 9},
        'oft': {'20170724': 105, '20170725': 105,
                '20180516': 94, '20180923': 94, '20181202': 94,
                '20190508': 94, '20190511': 94, '20190720': 102,
                '20200614': 135, '20200617': 135,
                '20210525': 135, '20210620': 135, '20210629': 135,
                '20210713': 135, '20210714': 135},
        }
pdpOdmmy = {dwdrs: {'20170724': None} for dwdrs in dwd_sites.values()
            if dwdrs not in pdpO}
pdpO = pdpO | pdpOdmmy
# =============================================================================
# NOISE LEVEL PRESET
cdwd_defaultnl = (36, 42, 0.1)
nlvl = {'bxp': (24, 29, 0.1),
        'jxp': (29, 33, 0.1),
        'axp': (15, 20, 0.1)}
nlvldmmy = {dwdrs: cdwd_defaultnl for dwdrs in dwd_sites.values()
            if dwdrs not in nlvl}
nlvl = nlvl | nlvldmmy
# =============================================================================
# MELTING LAYER HEIGHTS (DEFAULT)
mlvl = {'bxp': 3.0, 'jxp': 3.0, 'axp': 3.0}  # 2018/2019/20210714
mlvldmmy = {dwdrs: 2.0 for dwdrs in dwd_sites.values() if dwdrs not in mlvl}
mlvl = mlvl | mlvldmmy

mlyr_thk = {'bxp': 0.8, 'jxp': 0.8, 'axp': 0.8}  # 2018/2019/20210714
mlyr_thkdmmy = {dwdrs: 0.8 for dwdrs in dwd_sites.values()
                if dwdrs not in mlyr_thk}
mlyr_thk = mlyr_thk | mlyr_thkdmmy

# =============================================================================
# Definition dict
# =============================================================================
RPARAMS = [
    {'site_name': 'Boxpol', 'rband': 'X', 'elev': 'n_ppi_020deg',
     'rhvtc': rhvtbxp, 'signvel': -1, 'signpdp': 1, 'bclass': 205,
     'zdr_offset': 0., 'zh_offset': zhO['bxp'], 'phidp_prst': pdpO['bxp'],
     'mlt': mlvl['bxp'], 'mlk': mlyr_thk['bxp'],
     'rhvmin': rhohv_minx, 'nlvl': nlvl['bxp'],
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
     'rhvtc': rhvtjxp, 'signvel': -1, 'signpdp': 1,  'bclass': 143,
     'zdr_offset': 1.5, 'zh_offset': zhO['jxp'], 'phidp_prst': pdpO['jxp'],
     'mlk': mlyr_thk['jxp'], 'mlt': mlvl['jxp'],
     'rhvmin': rhohv_minx, 'nlvl': nlvl['jxp'],
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
    {'site_name': 'Aaxpol', 'rband': 'X', 'elev': 'sweep_4',
     'rhvtc': rhvtaxp, 'signvel': -1, 'signpdp': 1,  'bclass': 23,
     'zdr_offset': 1.5, 'zh_offset': zhO['axp'], 'phidp_prst': pdpO['axp'],
     'mlk': mlyr_thk['axp'], 'mlt': mlvl['axp'],
     'rhvmin': rhohv_minx, 'nlvl': nlvl['axp'],
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
    ]

rparamsdmmy = [
    {'site_name': dwdrsk, 'rband': 'C', 'elev': dflt_selevc,
     'rhvtc': rhvtc, 'signvel': 1, 'signpdp': 1, 'bclass': 207,
     'zdr_offset': 0, 'phidp_prst': pdpO.get(dwdrsv), 'zh_offset': 0,
     'mlk': mlyr_thk.get(dwdrsv), 'mlt': mlvl.get(dwdrsv),
     'rhvmin': rhohv_minc, 'nlvl': nlvl.get(dwdrsv),
     'wu_pdp': 1, 'kdpwl': 3, 'beta': att_betac, 'alphabetar': alphabetar_c,
     'rhvatt': rhvzdrattc_c,  # 'zhah_a': zhah_ac, 'zhah_b': zhah_bc,
     'rz_a': rz_ac, 'rz_b': rz_bc, 'rkdp_a': rkdp_ac, 'rkdp_b': rkdp_bc,
     'rzhzdr_a': rzhzdr_ac, 'rzhzdr_b': rzhzdr_bc, 'rzhzdr_c': rzhzdr_cc,
     'rkdpzdr_a': rkdpzdr_ac, 'rkdpzdr_b': rkdpzdr_bc, 'rkdpzdr_c': rkdpzdr_cc,
     'rz_haila': rz_hailac, 'rz_hailb': rz_hailbc, 'thr_zwsnw': thr_zwsnwc,
     'thr_zhail': thr_zhailc, 'f_rz_ml': f_rz_mlc, 'f_rz_sp': f_rz_spc,
     'ahkdp_a': ahkdp_ac, 'ahkdp_b': ahkdp_bc, 'adpkdp_a': adpkdp_ac,
     'adpkdp_b': adpkdp_bc, 'zdrzh_a': zdrzh_ac, 'zdrzh_b': zdrzh_bc}
    for dwdrsk, dwdrsv in dwd_sites.items() if dwdrsv not in RPARAMS]

RPARAMS = RPARAMS + rparamsdmmy

RPRODSLTX = {'r_adp': '$R(A_{DP})$',
             'r_ah': '$R(A_{H})$',
             'r_kdp': '$R(K_{DP})$',
             'r_z': '$R(Z_H)$',
             'r_ah_kdp': '$R(A_{H}) & R(K_{DP})$',
             'r_kdp_zdr': '$R(K_{DP}, Z_{DR})$',
             'r_z_ah': '$R(Z_H) & R(A_{H})$',
             'r_z_kdp': '$R(Z_{H}) & R(K_{DP})$',
             'r_z_zdr': '$R(Z_{H}, Z_{DR})$',
             'r_kdpopt': '$R(K_{DP})[opt]$',
             'r_zopt': '$R(Z_{H})[opt]$',
             'r_ah_kdpopt': '$R(A_{H}) & R(K_{DP})[opt]$',
             'r_zopt_ah': '$R(Z_{H})[opt] & R(A_{H})$',
             'r_zopt_kdp': '$R(Z_{H})[opt] & R(K_{DP})$',
             'r_zopt_kdpopt': '$R(Z_{H})[opt] & R(K_{DP})[opt]$',
             'r_aho_kdpo': '$R(A_{H}) & R(K_{DP})[evnt-spcf]$',
             'r_kdpo': '$R(K_{DP})[evnt-spcf]$',
             'r_zo': '$R(Z_{H})[evnt-spcf]$',
             'r_zo_ah': '$R(Z_{H}) & R(A_{H})[evnt-spcf]$',
             'r_zo_kdp': '$R(Z_{H}) & R(K_{DP})[evnt-spcf]$',
             'r_zo_zdr': '$R(Z_{H}, Z_{DR})[evnt-spcf]$'}
