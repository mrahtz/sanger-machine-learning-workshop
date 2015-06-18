#!/usr/bin/env python

"""
Read an EKG trace from the format provided by PhysioNet: 16-bit signed samples
"""

from __future__ import print_function
import numpy as np
import struct
import matplotlib.pyplot as plt

class Trace(object):
    """
    Helper methods for reading traces.
    """

    @staticmethod
    def read_ekg_data(input_file, scale):
        """
        Read and parse the EKG data in the given file.
        """
        with open(input_file, 'rb') as input_file:
            data_raw = input_file.read()
        len_bytes = len(data_raw)
        len_shorts = len_bytes/2
        unpack_string = '%dh' % len_shorts
        data_shorts = np.array(struct.unpack(unpack_string, data_raw))
        data_scaled = data_shorts * scale
        return data_scaled

    @staticmethod
    def plot_ekg(input_file, n_samples):
        """
        Plot the EKG data in the given file, for debugging.
        """
        ekg_data = Trace.read_ekg_data(input_file, scale=1.0)
        plt.plot(ekg_data[0:n_samples])
        plt.show()
