## Sanger Anomaly Detection Workshop Code

This repository contains code used for the unsupervised learning section of the
machine learning workshop given to the Systems group at Sanger.

The idea is based on Chapter 4, *More Complex, Adaptive Models* from
[Practical Machine Learning](https://www.safaribooksonline.com/library/view/practical-machine-learning/9781491914151/ch04.html)
by Ted Dunning and Ellen Friedman.

**Update**: Majid al-Dosari (in the comments at
<http://amid.fish/anomaly-detection-with-k-means-clustering>) and Eamonn Keogh
point out that there may be issues with the approach described here for the
reasons outlined in
[Clustering of Time Series Subsequences is Meaningless](http://www.cs.ucr.edu/~eamonn/meaningless.pdf).
This material still serves as an introduction to unsupervised learning and
clustering, but **beware in using it for anomaly detection in practice**.

### Contents

* `Unsupervised Learning.ipynb` is an IPython notebook demonstrating a simple example of unsupervised learning: time-series anomaly detection. View a static version of the notebook at http://nbviewer.ipython.org/github/mrahtz/sanger-machine-learning-workshop/blob/master/Unsupervised%20Learning.ipynb.
* `a02.dat` is a set of EKG data from PhysioNet used to demonstrate the
  algorithms.
* `ekg_data.py` is a module for reading the EKG data.
* `learn_utils.py` is a collection of helper functions developed in the
  notebook, saved as a module for reuse.
* `learn.py` is a complete listing for the code developed in the notebook.

### Requirements

Python is required, along with the following modules:
* NumPy
* matplotlib
* scikit-learn

IPython Notebook dependencies are also required, if running the notebook.

If you're on Ubuntu:
```
$ sudo apt-get install ipython-notebook python-numpy python-matplotlib python-sklearn
```
Or on any system with pip:
```
$ pip install ipython[notebook] numpy matplotlib scikit-learn
```
