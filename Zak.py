# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:42:53 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class Zak():
    """A class for calculating the Zak topological invariant for a given periodic
    superconductor S with parameters other than k_x and k_y."""
    def __init__(self, S, superconductor_params):
        self.S = S
        self.superconductor_params = superconductor_params
    def get_eigenstates(self, k_x, k_y):
        """Returns the eigenstates of the Hamiltonian in colunms."""
        H = self.S(k_x, k_y, **self.superconductor_params).matrix
        eigenvalues, eigenvectors = np.linalg.eigh(H)   #eigh gives always a real first element
        # for i in range(4):
        #     if eigenvectors[0, i] < 0:      #Assure the first element is positive real
        #         eigenvectors[:, i] *= -1
        return eigenvectors
    # def get_negative_energy_eigenstates(self, k_x, k_y):
    #     """Returns the eigenstates of the Hamiltonian in colunms."""
    #     H = self.S(k_x, k_y, **self.superconductor_params).matrix
    #     eigenvalues, eigenvectors = np.linalg.eigh(H)   #eigh gives always a real first element
    #     eigenvectors_negative = []
    #     for 
    #     return eigenvectors
    def get_eigenvalues(self, k_x, k_y):
        """Returns the eigenstates of the Hamiltonian in colunms."""
        H = self.S(k_x, k_y, **self.superconductor_params).matrix
        eigenvalues = np.linalg.eigvalsh(H)   
        return eigenvalues
    def _get_matrix_of_eigenstates(self, k_y, L_x):
        """Returns an array U with eigenvectors in columns for each k_x and a
        given k_y."""
        k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
        U = np.zeros((L_x, 4, 4), dtype=complex)
        for i, k_x in enumerate(k_x_values):
            U[i, :, :] = self.get_eigenstates(k_x, k_y)
        return U
    def _get_matrix_of_eigenstates_with_negative_energy(self, k_y, L_x):
        """Returns an array U with eigenvectors in columns for each k_x and a
        given k_y."""
        k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
        U = np.zeros((L_x, 4, 4), dtype=complex)
        for i, k_x in enumerate(k_x_values):
            U[i, :, :] = self.get_eigenstates(k_x, k_y)
        return U
    def get_Zak_Berry_phase(self, k_y, L_x):
        """Returns an array with the Berry phase for the four bands for a given
        k_y and discretization L_x."""
        U = self._get_matrix_of_eigenstates(k_y, L_x)
        derivative = np.diff(U, axis=0)
        sumand = np.zeros((L_x-1, 4), dtype=complex)
        for i in range(L_x-1):
            sumand[i, :] = np.diag(U[i, :, :].conj().T @ derivative[i, :, :])
        gamma = -np.imag(np.sum(sumand, axis=0))
        return gamma
    def get_Zak_log_phase(self, k_y, L_x):
        r"""
            Returns the Berry phase for a given k_y in a system of length L_x.
        .. math ::
            \gamma = - Im(ln(P)) \\
            P = <u_1 | u_2> <u_2|u_3> ... <u_{M-1} |u_1>
            """
        U = self._get_matrix_of_eigenstates(k_y, L_x)
        P = np.zeros((L_x, 4), dtype=complex)
        for i in range(L_x-1):
            P[i, :] = np.diag(U[i, :, :].conj().T @ U[i+1, :, :])
        P[L_x-1, :] = np.diag(U[L_x-1, :, :].conj().T @ U[0, :, :])
        gamma = -np.imag(np.log(np.prod(P, axis=0)))
        gamma = np.array([self._round_2pi_to_0(g) for g in gamma])
        return gamma
    def get_Berry_connection(self, k_y, L_x):
        """Returns the Berry connection for a given k_y and L_x.
        """
        A = np.zeros((L_x-1, 4, 4), dtype=complex)
        U = self._get_matrix_of_eigenstates(k_y, L_x)
        derivative = np.diff(U, axis=0)
        for i in range(L_x-1):
            A[i, :, :] = 1j * U[i, :, :].conj().T @ derivative[i, :, :]
        return A
    def get_Wilson_loop_matrix(self, k_y, L_x):
        """Returns the Wilson for a given
        k_y and discretization L_x where we consider only the occupied bands."""
        U = self._get_matrix_of_eigenstates(k_y, L_x)
        M = np.zeros((L_x, 2, 2), dtype=complex)
        for i in range(L_x):
            for j in range(2):
                for k in range(2):
                    if i==L_x-1:
                        M[i, j, k] = np.dot(U[i, :, :].conj().T[j, :], U[0, :, :][:, k])
                    else:
                        M[i, j, k] = np.dot(U[i, :, :].conj().T[j, :], U[i+1, :, :][:, k])
        W = np.linalg.multi_dot(M)
        return W
    def _round_2pi_to_0(self, value):
        mod_value = value % (2*np.pi)
        if np.isclose(mod_value, 0, rtol=1e-1) or np.isclose(mod_value,
                                                             2*np.pi, rtol=1e-1):
            return 0
        else:
            return mod_value
    def get_Wilson_spectrum(self, k_y, L_x):
        W = self.get_Wilson_loop_matrix(k_y, L_x)
        theta_n = -np.imag( np.log( np.linalg.eigvals(W) ) )
        theta_n = np.array([self._round_2pi_to_0(t) for t in theta_n])
        return theta_n