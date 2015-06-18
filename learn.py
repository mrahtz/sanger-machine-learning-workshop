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

WINDOW_LEN = 32

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
    Populate a list of all segments seen in the input data.
    Window using a half-sine function so that the resulting segments
    can be added together even if slightly overlapping, enabling
    later reconstruction.
    """
    n_samples = len(data)
    step = 2
    segments = []
    for segment in sliding_chunker(data, len(window), step):
        segment *= window
        # normalize: make the vector formed by the data unit length
        vector_length = np.linalg.norm(segment)
        segment /= vector_length
        segments.append(segment)

    return segments

def reconstruct(data, window, clusterer):
    """
    Reconstruct the given data using the cluster centers from the given
    clusterer.
    """
    chunks = \
        sliding_chunker(data, window_len=WINDOW_LEN, slide_len=WINDOW_LEN/2)
    reconstructed_data = np.zeros(len(data))
    for chunk_n, chunk in enumerate(chunks):
        # normalize and window the chunk so that we can find it in
        # our clusters...
        chunk *= window
        chunk_size = np.linalg.norm(chunk)
        chunk /= chunk_size
        nearest_match_idx = clusterer.predict(chunk)[0]
        nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])
        # ...then re-scale the reference by the same size so it matches
        # the chunk we're looking for
        nearest_match *= chunk_size

        pos = chunk_n * WINDOW_LEN/2
        reconstructed_data[pos:pos+WINDOW_LEN] += nearest_match

    return reconstructed_data


def main():
    n_samples = 1000
    print("Reading data...")
    data = ekg_data.read_ekg_data('a02.dat')[0:n_samples]

    window_rads = np.linspace(0, np.pi, WINDOW_LEN)
    window = np.sin(window_rads)**2

    print("Windowing data...")
    segments = get_windowed_segments(data, window)

    print("Clustering...")
    clusterer = KMeans(n_clusters=30)
    clusterer.fit(segments)

    print("Reconstructing...")
    reconstructed_data = reconstruct(data, window, clusterer)

    plt.figure()
    plt.plot(data[0:n_samples], label="Original EKG")
    plt.plot(reconstructed_data[0:n_samples], label="Reconstructed EKG")
    plt.legend()
    plt.show()

main()
