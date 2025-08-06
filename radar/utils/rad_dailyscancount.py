#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:04:53 2025

@author: dsanchez
"""

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
import seaborn as sns
from radar import twpext as tpx
import pickle


START_TIMES = [dt.datetime(2017, 7, 24, 0, 0),  # 24h [NO JXP]
               dt.datetime(2017, 7, 25, 0, 0),  # 24h [NO JXP]
               dt.datetime(2018, 5, 16, 0, 0),  # 24h []
               dt.datetime(2018, 9, 23, 0, 0),  # 24h [NO JXP]
               dt.datetime(2018, 12, 2, 0, 0),  # 24 h [NO JXP]
               dt.datetime(2019, 5, 8, 0, 0),   # 24h [NO JXP]
               dt.datetime(2019, 7, 20, 0, 0),  # 16h [NO BXP]
               dt.datetime(2020, 6, 17, 0, 0),  # 24h [NO BXP]
               dt.datetime(2021, 7, 13, 0, 0),  # 24h [NO BXP]
               dt.datetime(2021, 7, 14, 0, 0),  # 24h [NO BXP]
               ]

STOP_TIMES = [i+dt.timedelta(hours=24) for i in START_TIMES]

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

RES_DIR = LWDIR + 'pd_rdres/qpe_all/'

SAVE_FIGS = False

# =============================================================================
# Define radar sites and list files
# =============================================================================
RSITES = ['Boxpol', 'Juxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach',
          'Offenthal']

# RSITE_FILES = {
#     i: {dts[0].strftime("%Y%m%d"):
#         tpx.get_listfilesxpol(i, dts[0], dts[1],
#                               ('n_ppi_010deg' if i == 'Boxpol' else 'sweep_9'))
#         if 'xpol' in i.lower() else
#         tpx.get_listfilesdwd(i, dts[0], dts[1], 'ppi_pcp')
#         for dts in zip(START_TIMES, STOP_TIMES)}
#     for i in RSITES}

# with open(RES_DIR + f'dwdxpol_{len(START_TIMES)}e_flist.tpy', 'wb') as f:
#     pickle.dump(RSITE_FILES, f, pickle.HIGHEST_PROTOCOL)

with open(RES_DIR + f'dwdxpol_{len(START_TIMES)}e_flist.tpy', 'rb') as breader:
    lfiles = pickle.load(breader)

lfiles['Boxpol']['20210713'] = []
lfiles['Boxpol']['20210714'] = []
# %%

rs_dtms = {rs:
           {dt1: [dt.datetime.strptime(i[i.find(
               ('12345_' if rs == 'Boxpol' else '99999_'))
               + 6:].removesuffix('_00.h5'), '%Y%m%d%H%M%S') for i in lf1]
            if 'xpol' in rs
            else [dt.datetime.strptime(i[0].removesuffix(
                '-hd5')[i[0].find('_00-') + 4:-10], '%Y%m%d%H%M%S%f')
                for i in lf1]
            for dt1, lf1 in lfiles[rs].items()} for rs in RSITES}

rs_dtmsf = {rs: np.hstack([idt for k1, idt in rs_dtms[rs].items()])
            for rs in RSITES}

# %%

results = {rs: [len(ied) for k1, ied in rs_dtms[rs].items()]
           for rs in RSITES}

# c = ['#99BAB9', '#B3E1F8', '#716C9F', '#F2EAC4', '#F6D8A6', '#F3B1A7']
# cl = {rs: c[cnt] for cnt, rs in enumerate(RSITES)}
# category_colors = plt.colormaps['Set3'](np.linspace(0.35, 0.8, 6))
# category_colors = plt.colormaps['Set3'](np.linspace(0., 1, 6))
category_colors = plt.colormaps['Dark2'](np.linspace(0., 1, 6))
cl = {rs: category_colors[cnt] for cnt, rs in enumerate(RSITES)}

width = 0.5
dtevents = [f'{x:%Y-%m-%d}' for x in START_TIMES]
nscans_rs = {k1: np.array(v1) for k1, v1 in results.items()}
# plt.style.use('seaborn-v0_8-whitegrid')
plt.style.use('default')
fig, ax = plt.subplots(figsize=(15, 3))
fig.suptitle("Daily scan count for each radar site", fontsize=16)
bottom = np.zeros(len(dtevents))

for rsname, weight_count in nscans_rs.items():
    if rsname == 'Boxpol':
        rsname_m = 'BoXPol'
    elif rsname == 'Juxpol':
        rsname_m = 'JuXPol'
    else:
        rsname_m = rsname
    pbar = ax.bar(dtevents, weight_count, width, label=rsname_m, bottom=bottom,
                  # color=category_colors,
                  color=cl[rsname]
                  # colormap="GnBu",
                  )
    labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in pbar]
    ax.bar_label(pbar, labels=labels, label_type='center', color='k')
    bottom += weight_count

ax.grid(axis='both', which='both')
ax.set_axisbelow(True)
ax.legend(ncols=len(START_TIMES), bbox_to_anchor=(0.5, 1.3),
          loc='upper center', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

# ax.set_title("Number of daily scans per radar site")
plt.tight_layout()
fig.subplots_adjust(top=0.72)
plt.show()

if SAVE_FIGS:
    len_rdtsets = len(START_TIMES)
    RES_DIR2 = RES_DIR + 'rcomp_qpe_dwd_dwdxpol/'
    fname = (f"dailyscans_{len_rdtsets}_events.png")
    plt.savefig(RES_DIR2 + fname, dpi=200, format='png')
# %%

# fig, ax = plt.subplots(figsize=(15, 2))
# for cnt, rs in enumerate(RSITES):
#     ax.plot(rs_dtmsf[rs], np.ones_like(rs_dtmsf[rs])*cnt, 'x-')
# %%

# rs = 'Boxpol'

# fig, ax = plt.subplots(figsize=(15,5))
# for cnt, rs in enumerate(reversed(RSITES)):
#     x_values = [f'{x:%Y-%m-%d}' for x in rs_dtmsf[rs]]
#     ax.bar(x_values, np.ones_like(rs_dtmsf[rs])*cnt)
# plt.show()
# %%

# category_names = START_TIMES
# results = {
#     rs: [len(ied) for k1, ied in rs_dtms[rs].items()]
#     for rs in RSITES}


# def survey(results, category_names):
#     """
#     Parameters
#     ----------
#     results : dict
#         A mapping from question labels to a list of answers per category.
#         It is assumed all lists contain the same number of entries and that
#         it matches the length of *category_names*.
#     category_names : list of str
#         The category labels.
#     """
#     labels = list(results.keys())
#     data = np.array(list(results.values()))
#     data_cum = data.cumsum(axis=1)
#     # data_cum = 2880
#     category_colors = plt.colormaps['viridis_r'](
#         np.linspace(0.35, 0.8, data.shape[1]))

#     fig, ax = plt.subplots(figsize=(15, 5))
#     ax.invert_yaxis()
#     ax.xaxis.set_visible(False)
#     ax.set_xlim(0, np.sum(data, axis=1).max())

#     for i, (colname, color) in enumerate(zip(category_names, category_colors)):
#         widths = data[:, i]
#         starts = data_cum[:, i] - widths
#         rects = ax.barh(labels, widths, left=starts, height=0.5,
#                         label=colname, color=color)

#         r, g, b, _ = color
#         text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#         ax.bar_label(rects, label_type='center', color=text_color)
#     ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
#               loc='lower left', fontsize='small')

#     return fig, ax


# survey(results, category_names)
# # ax.tight_layout()

# plt.show()

# data = [    (dt.datetime(2018, 7, 17, 0, 15), dt.datetime(2018, 7, 17, 0, 30), 'Boxpol'),
#             (dt.datetime(2018, 7, 17, 0, 30), dt.datetime(2018, 7, 17, 0, 45), 'Juxpol'),
#             (dt.datetime(2018, 7, 17, 0, 45), dt.datetime(2018, 7, 17, 1, 0), 'work'),
#             (dt.datetime(2018, 7, 17, 1, 0), dt.datetime(2018, 7, 17, 1, 30), 'Boxpol'),
#             (dt.datetime(2018, 7, 17, 1, 15), dt.datetime(2018, 7, 17, 1, 30), 'Juxpol'), 
#             (dt.datetime(2018, 7, 17, 1, 30), dt.datetime(2018, 7, 17, 1, 45), 'work')
#         ]

# cats = {"Boxpol" : 1, "Juxpol" : 2, "work" : 3}
# colormapping = {"Boxpol" : "C0", "Juxpol" : "C1", "work" : "C2"}

# verts = []
# colors = []
# for d in data:
#     v =  [(mdates.date2num(d[0]), cats[d[2]]-.25),
#           (mdates.date2num(d[0]), cats[d[2]]+.25),
#           (mdates.date2num(d[1]), cats[d[2]]+.25),
#           (mdates.date2num(d[1]), cats[d[2]]-.25),
#           (mdates.date2num(d[0]), cats[d[2]]-.25)]
#     verts.append(v)
#     colors.append(colormapping[d[2]])

# bars = PolyCollection(verts, facecolors=colors,# mutation_aspect=0.2
#                       )

# fig, ax = plt.subplots(figsize=(15, 2))
# ax.add_collection(bars)
# ax.autoscale()
# # ax.xaxis.set_major_formatter(
# #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# # ax.xaxis.set_minor_locator(mdates.MonthLocator())
# ax.grid(True)
# # loc = mdates.MinuteLocator(byminute=[0,15,30,45])
# # ax.xaxis.set_major_locator(loc)
# # ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

# ax.set_yticks([1,2,3])
# ax.set_yticklabels(["Boxpol", "Juxpol", "work"])
# plt.tight_layout()

# plt.show()

# %%


# mydict = {
#     'Event': ['Running', 'Swimming', 'Biking', 'Hiking'],
#     'Completed': [2, 4, 3, 7],
#     'Participants': [10, 20, 35, 10]}

# df = pd.DataFrame(mydict).set_index('Event')
# df = df.assign(Completion=(df.Completed / df.Participants) * 100)
# # print(df)


# plt.subplots(figsize=(5, 2))
# sns.set_color_codes("pastel")
# ax = sns.barplot(x=df.Completion, y=df.index, joinstyle='bevel')

# new_patches = []
# for patch in reversed(ax.patches):
#     # print(bb.xmin, bb.ymin,abs(bb.width), abs(bb.height))
#     bb = patch.get_bbox()
#     color = patch.get_facecolor()
#     p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
#                             abs(bb.width), abs(bb.height),
#                             boxstyle="round,pad=-0.0040,rounding_size=2",
#                             ec="none", fc=color,
#                             mutation_aspect=0.2
#                             )
#     patch.remove()
#     new_patches.append(p_bbox)

# for patch in new_patches:
#     ax.add_patch(patch)

# sns.despine(left=True, bottom=True)

# ax.tick_params(axis=u'both', which=u'both', length=0)
# plt.tight_layout()
# plt.show()
