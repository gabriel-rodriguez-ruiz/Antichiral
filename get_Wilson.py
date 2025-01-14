#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:56:20 2025

@author: gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKXKY
import numpy as np
from Zak import Zak
import matplotlib.pyplot as plt

k_y = 0.1
L_x = 100

t = 10   #10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B = 2*Delta_0       #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40  #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

Z = Zak(ZKMBSuperconductorKXKY, superconductor_params)

W = Z.get_Wilson_loop_matrix(k_y, L_x)
theta_n = -np.imag( np.log( np.linalg.eigvals(W) ) )

print(theta_n)