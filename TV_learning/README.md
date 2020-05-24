# Total variation computational learning

This code is an implementation of the Total variation model for supervised learning as described in eqns (11) and (12) in the paper:

Supervised Learning via Euler's Elastica Models
Tong Lin, Hanlin Xue, Ling Wang, Bo Huang, Hongbin Zha.
Year: 2015, Volume: 16, Issue: 111, Pages: 3637−3686

There is in the folder a self contained Jupyter notebook including all needed functions
and a Python 3 implementation consisting in three files.

Usage:

python TVL_main.py

In both, the Jupyter and Python implementations, there is the option of using automatic differentiation by choosing the appropriate functions to approximate the gradient, Laplacian and Hessian differential operators.

A dataset for liver disorders (bupaNormalized.dat) is included for testing. You may download the original files from https://archive.ics.uci.edu/ml/datasets/liver+disorders
