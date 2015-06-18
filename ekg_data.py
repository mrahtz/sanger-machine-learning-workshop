#!/usr/bin/env python

"""
Helper functions for reading EKG trace data
"""

from __future__ import print_function
import numpy as np
import struct
import matplotlib.pyplot as plt

def read_ekg_data(input_file):
    """
    Read the EKG data from the given file.
    """
    with open(input_file, 'rb') as input_file:
        data_raw = input_file.read()
    n_bytes = len(data_raw)
    n_shorts = n_bytes/2
    # '<': little-endian
    unpack_string = '<%dh' % n_shorts
    # sklearn seems to throw up if data not in float format
    data_shorts = np.array(struct.unpack(unpack_string, data_raw)).astype(float)
    return data_shorts

def plot_ekg(input_file, n_samples):
    """
    Plot the EKG data from the given file (for debugging).
    """
    ekg_data = Trace.read_ekg_data(input_file, scale=1.0)
    plt.plot(ekg_data[0:n_samples])
    plt.show()
