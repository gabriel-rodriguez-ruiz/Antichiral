#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:21:33 2025

@author: gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKXKY, ZKMBSuperconductorKY
import numpy as np
from Zak import Zak
import matplotlib.pyplot as plt

L_x = 200
L_y = 40
# k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
k_y_values = np.pi/L_y*np.arange(-L_y/2, L_y/2) / 5

t = 10   #10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0

B = 0.5*Delta_0    #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40  #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

Berry_B = np.zeros((len(k_y_values), 2))
E_k_y = np.zeros((len(k_y_values), 4*L_x))
E_0_k_y = np.zeros((len(k_y_values), 4))

for i, k_y in enumerate(k_y_values):
    Z = Zak(ZKMBSuperconductorKXKY, superconductor_params)
    Berry_B[i, :] = Z.get_Wilson_spectrum(k_y, L_x)
    S = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1, Lambda,
                               B_x, B_y, B_z)
    E_k_y[i, :] = np.linalg.eigvalsh(S.matrix)
    E_0_k_y[i, :] = Z.get_eigenvalues(0, k_y)   #k_x=0

#%%
plt.rc('font', size=24)          # controls default text sizes
plt.rcParams.update({
    "text.usetex": False})
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

axs[0].plot(k_y_values, E_k_y/Delta_0, color="black")
axs[0].plot(k_y_values[10:90], E_k_y[10:90,2*L_x-1:2*L_x+1]/Delta_0, color="red")

# axs[0].set_xlabel(r"$k_y$")
axs[0].set_ylabel(r"$\frac{E(k_y)}{\Delta_0}$", fontsize=30)
# axs[0].set_title(r"$\mu=$"+f"{np.round(S.mu, 2)}")
fig.suptitle(r"$L_x=$"+f"{L_x}"
             +r"; $\lambda=$" + f"{S.Lambda:.2}"
             +r"; $\Delta_0=$" + f"{S.Delta_0}"
             +r"; $\Delta_1=$" + f"{S.Delta_1}"
             +r"; $w_0=$"+f"{S.t}" + "\n"
             +r"$B_x=$"+f"{np.round(B_x, 2)}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $B_z=$"+f"{np.round(B_z, 2)}"
             +r"; $\mu=$"+f"{np.round(S.mu, 2)}")
axs[0].set_ylim((-2, 2))

axs[1].scatter(k_y_values, Berry_B[:, 0], label=r"$\theta_1$", linewidth=5)
axs[1].scatter(k_y_values, Berry_B[:, 1], label=r"$\theta_2$", linewidth=5)

axs[1].set_xlabel(r"$k_y$", fontsize=30)
axs[1].set_ylabel(r"$\theta_n$", fontsize=30)
axs[1].legend(fontsize= "xx-small")
# axs[1].set_title(r"$B_x=$" + f"{np.round(B_x, 2)}" + r"$; \mu=$" + f"{mu}" +
#              r"; $\lambda=$" + f"{Lambda}" + r"; $\Delta_0=$" + f"{Delta_0}"+
#              r"; $L_x=$" + f"{L_x}")



plt.tight_layout()
plt.show()
