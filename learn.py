#!/usr/bin/env python

"""
Read a bunch of EKG data, chop out windows and cluster the windows. Then
reconstruct the signal and figure out the error.
"""

from __future__ import print_function
import numpy as np
from trace import Trace

def main():
    scale = 1.0/200
    ekg_data = Trace.read_ekg_data('a02.dat', scale)

    Trace.plot_ekg('a02.dat', n_samples=1000)

main()
