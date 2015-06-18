#!/usr/bin/env python

"""
Read a bunch of EKG data, chop out windows and cluster the windows. Then
reconstruct the signal and figure out the error.
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from trace import Trace

def get_segments():
    """
    Populate a list of all segments seen in the input data.

    (This is done by applying a sliding window across the data,
    with a half-sine as the window function.)
    """
    scale = 1.0/200
    ekg_data = Trace.read_ekg_data('a02.dat', scale)

    window_size = 32
    window_rads = np.linspace(0, np.pi, window_size)
    window = np.sin(window_rads)**2

    n_samples = 10
    step = 2
    segments = []
    for sample_n in range(0, n_samples):
        offset = sample_n * step
        segment = ekg_data[offset:offset+window_size]
        segment *= window
        # normalize: make the vector formed by the data unit length
        vector_length = np.linalg.norm(segment)
        segment /= vector_length
        segments.append(segment)

    return segments

def main():
    segments = get_segments()
    for segment_n in range(5):
        plt.figure()
        plt.plot(segments[segment_n])
    plt.show()

main()
