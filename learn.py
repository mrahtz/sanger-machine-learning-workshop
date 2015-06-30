#!/usr/bin/env python

"""
Read a bunch of EKG data, chop out windows and cluster the windows. Then
reconstruct the signal and figure out the error.
"""

from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ekg_data
import learn_utils

WINDOW_LEN = 32

def get_windowed_segments(data, window):
    """
    Populate a list of all segments seen in the input data.  Apply a window to
    each segment so that they can be added together even if slightly
    overlapping, enabling later reconstruction.
    """
    step = 2
    windowed_segments = []
    segments = learn_utils.sliding_chunker(
        data,
        window_len=len(window),
        slide_len=step
    )
    for segment in segments:
        segment *= window
        windowed_segments.append(segment)
    return windowed_segments

def main():
    """
    Main function.
    """
    n_samples = 8192
    print("Reading data...")
    data = ekg_data.read_ekg_data('a02.dat')[0:n_samples]

    window_rads = np.linspace(0, np.pi, WINDOW_LEN)
    window = np.sin(window_rads)**2
    print("Windowing data...")
    windowed_segments = get_windowed_segments(data, window)

    print("Clustering...")
    clusterer = KMeans(n_clusters=150)
    clusterer.fit(windowed_segments)

    print("Reconstructing...")
    reconstruction = learn_utils.reconstruct(data, window, clusterer)
    error = reconstruction - data
    print("Maximum reconstruction error is %.1f" % max(error))

    plt.figure()
    n_plot_samples = 300
    plt.plot(data[0:n_plot_samples], label="Original EKG")
    plt.plot(reconstruction[0:n_plot_samples], label="Reconstructed EKG")
    plt.plot(error[0:n_plot_samples], label="Reconstruction error")
    plt.legend()
    plt.show()

main()
