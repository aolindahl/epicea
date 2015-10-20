# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:42:11 2015

@author: antlin
"""

def update_progress(i_evt, n_events, verbose=True):
    if (verbose and
            ((i_evt % (n_events // 100 if n_events > 100 else n_events // 10)
                == 0) or (i_evt == n_events-1))):
        progress = (100 * i_evt) // (n_events - 1)
        num_squares = 40
        base_string = '\r[{:' + str(num_squares) + '}] {}%'
        print(base_string.format('#' * (progress * num_squares // 100),
                                 progress),
              end='', flush=True)
