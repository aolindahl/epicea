# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:41:21 2015

@author: antlin
"""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

FONTSIZE = 10


def figure_wrapper(name):
    fig_handle = plt.figure(name, figsize=(11, 8))
    plt.clf()
    plt.suptitle(name)
    return fig_handle


def text_wrapper(function, text):
    function(text, fontsize=FONTSIZE)


def xlabel_wrapper(text):
    text_wrapper(plt.xlabel, text)


def ylabel_wrapper(text):
    text_wrapper(plt.ylabel, text)


def legend_wrapper(loc='best', ax=None):
    if ax is None:
        ax = plt
    ax.legend(loc=loc, fontsize=FONTSIZE)


def title_wrapper(text):
    plt.title(text, fontsize=FONTSIZE)


def colorbar_wrapper(label=None, mappable=None):
    cbar = plt.colorbar(mappable=mappable)
    cbar.ax.tick_params(labelsize=FONTSIZE)
    if label is not None:
        cbar.set_label(label)


def tick_fontsize(axis=None):
    if axis is None:
        axis = plt.gca()
    for xyaxis in ['xaxis', 'yaxis']:
        for tick in getattr(axis, xyaxis).get_major_ticks():
            tick.label.set_fontsize(FONTSIZE)


def bar_wrapper(x, y, color=None, label=None, verbose=False):
    if verbose:
        print 'In bar_wrapper()'
        print 'x.shape =', x.shape
        print 'y.shape =', y.shape
    width = np.diff(x).mean(dtype=float)
    plt.bar(x - width/2, y, width=width, linewidth=0, color=color,
            label=label)


def imshow_wrapper(img, x_centers, y_centers=None, ax=plt, kw_args={}):
    x_step = np.diff(x_centers).mean(dtype=float)
    x_min = x_centers.min() - x_step/2
    x_max = x_centers.max() + x_step/2
    if y_centers is None:
        y_min, y_max = x_min, x_max
    else:
        y_step = np.diff(y_centers).mean(dtype=float)
        y_min = y_centers.min() - y_step/2
        y_max = y_centers.max() + y_step/2

    axis = (x_min, x_max, y_min, y_max)
    img_ax = ax.imshow(img, extent=axis, origin='lower',
                    interpolation='none', **kw_args)
    ax.axis(axis)
    return img_ax


def savefig_wrapper():
    fig_path = 'first_look_figures'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig = plt.gcf()
    fig_name = fig.canvas.get_window_title()
    fig_name = fig_name.replace(' ', '_')
    file_name = '.'.join([fig_name, 'pdf'])
    plt.savefig(os.path.join(fig_path, file_name))
