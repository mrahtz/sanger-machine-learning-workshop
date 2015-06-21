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

WINDOW_LEN = 16

def sliding_chunker(data, window_len, slide_len):
    """
    Split a list into a series of sub-lists, each sub-list window_len long,
    sliding along by slide_len each time. If the list doesn't have enough
    elements for the final sub-list to be window_len long, the remaining data
    will be dropped.

    e.g. sliding_chunker(range(6), window_len=3, slide_len=2)
    gives [ [0, 1, 2], [2, 3, 4] ]
    """
    chunks = []
    for pos in range(0, len(data), slide_len):
        chunk = np.copy(data[pos:pos+window_len])
        if len(chunk) != window_len:
            continue
        chunks.append(chunk)

    return chunks

def get_windowed_segments(data, window):
    """
    Populate a list of all segments seen in the input data.  Apply a window to
    each segment so that they can be added together even if slightly
    overlapping, enabling later reconstruction.
    """
    step = 2
    windowed_segments = []
    segments = sliding_chunker(data, window_len=len(window), slide_len=step)
    for segment in segments:
        segment *= window
        # normalize: make the vector formed by the data in n-dimensional space
        # (where n is the number of elements in the vector) unit length
        vector_length = np.linalg.norm(segment)
        segment /= vector_length
        windowed_segments.append(segment)
    return windowed_segments

def reconstruct(data, window, clusterer):
    """
    Reconstruct the given data using the cluster centers from the given
    clusterer.
    """
    slide_len = WINDOW_LEN/2
    segments = \
        sliding_chunker(data, window_len=WINDOW_LEN, slide_len=slide_len)
    reconstructed_data = np.zeros(len(data))
    for segment_n, segment in enumerate(segments):
        # normalize and window the segment so that we can find it in
        # our clusters...
        segment *= window
        vector_length = np.linalg.norm(segment)
        segment /= vector_length
        nearest_match_idx = clusterer.predict(segment)[0]
        nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])
        # ...then re-scale the reference by the same size so it matches
        # the segment we're looking for
        nearest_match *= vector_length

        pos = segment_n * slide_len
        reconstructed_data[pos:pos+WINDOW_LEN] += nearest_match

    return reconstructed_data

def main():
    """
    Main function.
    """
    n_samples = 1000
    print("Reading data...")
    data = ekg_data.read_ekg_data('a02.dat')[0:n_samples]

    training_data = data[0:500]
    test_data = data[500:1000]

    window_rads = np.linspace(0, np.pi, WINDOW_LEN)
    window = np.sin(window_rads)**2

    print("Windowing data...")
    training_segments = get_windowed_segments(training_data, window)

    print("Clustering...")
    clusterer = KMeans(n_clusters=30)
    clusterer.fit(training_segments)

    print("Reconstructing...")
    reconstructed_test_data = reconstruct(test_data, window, clusterer)
    error = reconstructed_test_data - test_data
    print("Maximum reconstruction error is %.1f" % max(error))

    plt.figure()
    plt.plot(test_data, label="Original EKG")
    plt.plot(reconstructed_test_data, label="Reconstructed EKG")
    plt.plot(error, label="Reconstruction error")
    plt.legend()
    plt.show()

main()
