import numpy as np
import matplotlib.pyplot as plt

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

def plot_chunks(chunks, chunk_n_step):
    """
    Plot a set of 9 chunks from the given set, starting from the first one
    and increasing in index by chunk_n_step for each subsequent graph
    """
    plt.figure()
    n_graph_rows = 3
    n_graph_cols = 3
    graph_n = 1
    chunk_n = 0
    for row in range(n_graph_rows):
        for col in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            axes.set_ylim([-100, 150])
            plt.plot(chunks[chunk_n])
            graph_n += 1
            chunk_n += chunk_n_step
    # fix subplot sizes so that everything fits
    plt.tight_layout()
    plt.show()
