#!/usr/bin/env python

"""
Read a bunch of EKG data, chop out windows and cluster the windows. Then
reconstruct the signal and figure out the error.
"""

from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from trace import Trace

WINDOW_LEN = 32

def get_segments(ekg_data, window):
    """
    Populate a list of all segments seen in the input data.

    (This is done by applying a sliding window across the data,
    with a half-sine as the window function.)
    """

    n_samples = 512
    step = 2
    segments = []
    for sample_n in range(0, n_samples-WINDOW_LEN+1):
        offset = sample_n * step
        segment = np.copy(ekg_data[offset:offset+WINDOW_LEN])
        segment *= window
        # normalize: make the vector formed by the data unit length
        vector_length = np.linalg.norm(segment)
        segment /= vector_length
        segments.append(segment)

    return segments

def sliding_chunker(data, window_len, slide_len):
    """
    Split a list into a series of sub-lists, each sub-list window_len long,
    sliding along by slide_len each time.

    e.g. sliding_chunker(range(10), window_len=3, slide_len=2)
    gives [ [0, 1, 2], [2, 3, 4], ... ]
    """
    chunks = []
    for pos in range(0, len(data), slide_len):
        chunk = np.copy(data[pos:pos+window_len])
        chunks.append(chunk)
    return chunks

def main():
    scale = 1.0/200
    print("Reading data...")
    ekg_data = Trace.read_ekg_data('a02.dat', scale)[0:1000]

    window_rads = np.linspace(0, np.pi, WINDOW_LEN)
    window = np.sin(window_rads)**2

    print("Windowing data...")
    segments = get_segments(ekg_data, window)

    print("Clustering...")
    clusterer = KMeans(n_clusters=30)
    clusterer.fit(segments)

    print("Reconstructing...")
    reconstructed_ekg_data = np.zeros(len(ekg_data))
    for chunk_n, chunk in enumerate(sliding_chunker(ekg_data,
            window_len=WINDOW_LEN, slide_len=WINDOW_LEN/2)):

        pos = chunk_n*WINDOW_LEN/2
        if (pos + WINDOW_LEN) > len(ekg_data):
            break

        chunk *= window
        chunk_size = np.linalg.norm(chunk)
        # normalize the chunk so that we can actually find a reference...
        chunk /= chunk_size
        nearest_match_idx = clusterer.predict(chunk)[0]
        nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])
        # ...then re-scale the reference by the same size so it matches
        # the chunk we're looking for
        nearest_match *= chunk_size

        reconstructed_ekg_data[pos:pos+WINDOW_LEN] += nearest_match

    plt.figure()
    plt.plot(ekg_data[0:500], label="Original EKG")
    plt.plot(reconstructed_ekg_data[0:500], label="Reconstructed EKG")
    plt.legend()
    plt.show()

main()
