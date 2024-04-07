import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import mathieu_a
import scipy.constants as sp


class Qubit:
    def __init__(self, Ej  = 50.0, Ec = 1.0, ng = 0.5, n=15, phi_shift=0, phi_limit=np.pi/2):
        """
            Ej (float): Josephson energy, in GHz
            Ec (float): Charging energy of the CPB, in GHz
            ng (float): Offset charge
            n (int): number of energy levels to be calculated
            phi_shift (float): shift in phase to be applied when calculating Hamiltonian
            phi_limit (float): limits of sweep when calculating values using phi as independent variable
        """

        # initialize vals
        self.Ej = Ej
        self.Ec = Ec
        self.ng = ng # evaluate around sweet spot 0.5 unless otherwise specified
        self.n = n # number of energy levels to calculate
        self.eigenvalues = None
        self.eigenvectors = None
        self.phi_shift = phi_shift
        self.phi_limit = phi_limit
        
        if n > 0:
            self.populate_eigen()

    def populate_eigen(self):
        # Generate the diagonal and offdiagonal components of the Hamiltonian
        self.eigenvalues, self.eigenvectors = self.solve_H()

        
    def solve_H(self, ng=None, dep="phi", basis = "nhat", drive = False):
        """
            ng (float): Offset charge point to be evaluated at
        """
        # compute the eigenvectors and eigenvalues of the CPB
        # all properties can be derived from these

        if ng is None:
            ng = self.ng

        # avoid diagonalizing without vals
        if (self.Ej is None) or (self.Ec is None):
            self.eigenvalues, self.eigenvectors = None, None
            return None

        else:
            if drive:
                diag = np.arange(-self.n, self.n)
                h_diag = sp.hbar

            # If using the nhat (number of Cooper Pairs) as basis
            if dep=="phi" and basis == "nhat":
                diag = np.arange(-self.n, self.n+1)
                # phi_0 = sp.h / (2* np.e)
                
                # Ej_mod = self.Ej * np.abs(np.cos(np.pi * (self.phi_shift) / phi_0))

                phi = np.linspace(-self.phi_limit , self.phi_limit, self.n*2)
                h_diag = 4 * self.Ec * (np.ones( 2* self.n + 1) - ng)**2
                h_off = -(self.Ej/2.0) * np.cos(phi + self.phi_shift) * np.ones(len(diag)-1)

                # get eigenvalues, eigenvectors
                eigenvalues, evecs = linalg.eigh_tridiagonal(h_diag, h_off)
                return np.real(np.array(eigenvalues)), self.normalize_evecs(np.array(evecs))

  
            
            # If using phi
            elif dep == "phi" and basis == "phi":
                diag = np.arange(-self.n, self.n)

                phi = np.linspace(-self.phi_limit , self.phi_limit, self.n*2 + 1)
                h_off = (4 * self.Ec * (np.ones( 2* self.n) - ng)**2) * .5
                h_diag = -(self.Ej) * np.cos(phi + self.phi_shift)

                # get eigenvalues, eigenvectors
                eigenvalues, evecs = linalg.eigh_tridiagonal(h_diag, h_off)
                return np.real(np.array(eigenvalues)), self.normalize_evecs(np.array(evecs))

            elif dep =="n_hat":
                diag = np.arange(-self.n, self.n+1)
                
                h_diag = 4 * self.Ec * (diag - ng)**2
                h_off = -(self.Ej/2.0) * np.cos(self.phi_shift) * np.ones(len(diag)-1)

                # get eigenvalues, eigenvectors
                eigenvalues, evecs = linalg.eigh_tridiagonal(h_diag, h_off)
                return np.real(np.array(eigenvalues)), self.normalize_evecs(np.array(evecs))

    def normalize_eigenvalues(self, eigenvalues):
        """
        Normalize eigenvalues outputted from linalg.eigh_tridiagonal.

        Parameters:
            eigenvalues (1D array): Array of eigenvalues, where each index corresponds to the eigenvalue of that energy level.
        """
        normalized_eigenvalues = []

        ground_state_min = min(eigenvalues[0])
        for m in range(self.n):
            normalized_level = np.array(eigenvalues[m]) - ground_state_min
            normalized_eigenvalues.append(normalized_level)

        eval, _ = self.solve_H(ng=0.5, dep="n_hat")
        E01 = self.get_E01(ng=0.5)
        E01N = eval[1]-eval[0]

        for m in range(self.n):
            if self.n >= 1:
                normalized_eigenvalues[m] = normalized_eigenvalues[m] / E01N

        return normalized_eigenvalues
    
    def normalize_evecs(self, eigenvectors):
        """
        Normalize eigenvectors outputted from linalg.eigh_tridiagonal.

        Parameters:
            eigenvectors (2D array): Array of eigenvectors, each of which is an array of values, 
                                    where each index of the array corresponds to the eigenvector of that energy level.
        """
        copy = eigenvectors

        for i in range(len(eigenvectors)):
            evec = eigenvectors[:,i]
            phi = np.linspace(-self.phi_limit, self.phi_limit, len(evec))
            delta_phi = phi[1] - phi[0]

            norm = np.sqrt(delta_phi * sum(evec**2))
            copy[:,i] = evec / norm
        return copy


    def get_E01(self, ng):
        """
            ng (float): Offset charge point to be evaluated at
        """
        #energies are given in units of transition energy, E_01, evaluated at the degeneracy point ng = .5
        self.E01 = np.sqrt((((4*self.Ec * (2*ng - 1)) ** 2) + self.Ej ** 2)) # transition energy
        
        return self.E01

    def sweep_ng(self, sweep = np.linspace(-np.pi, np.pi, 201)):
        """
            dep (str): dependent variable
            m (integer): energy level
            ng_sweep (list[float]): Offset charge to be swept
        """

        eigenvalues = [[] for _ in range(self.n)]
        eigenvectors = [[] for _ in range(self.n)]
        
        for ng in sweep:
            evals, evecs = self.solve_H(ng, dep="n_hat")
            for m in range(self.n):
                eigenvalues[m].append((evals[m]))
                eigenvectors[m].append((evecs[:, m]))

        return self.normalize_eigenvalues(eigenvalues), eigenvectors                


    def sweep_phi(self, m=0, sweep = np.linspace(-np.pi, np.pi, 201)):
        # for plotting wrt phi
        eigenvalues, evecs = self.solve_H(ng=0.5)

        evec = evecs[:, m]
        n = np.arange(-self.n, self.n + 1)
        psi = []
        for i, val in enumerate(n):
            # Get Fourier component of each charge basis state
            psi.append(evec[i] * np.exp(1j * val * sweep))
        psi = np.array(psi)
        # Sum over Fourier components to get eigenwave
        psi = np.sum(psi, axis=0) / np.sqrt(2 * np.pi)
        # Normalize Psi
        norm = np.sqrt(np.dot(psi, psi.conj()))
        psi = psi / norm
        return psi, sweep
        

    def get_eigenvalues(self):
        return self.eigenvalues


    def get_Ej(self):
        return self.Ej

    def print_variables(self):
        print(f"n: {self.n}")
        print(f"Ej: {self.Ej}")
        print(f"Ec: {self.Ec}")
        print(f"ng: {self.ng}")
        print(f"eigenvalues: {self.eigenvalues}")
        print(f"eigenvectors: {self.eigenvectors}")
        print(f"E01: {self.E01}")

def plot_EjEcVSng(En, ng = np.linspace(-2.0,2.0,101)):
        """
        Plot the four diagrams showing how Ej/Ec affects the charge noise sensitivity of the qubit.

        Parameters:
            En (2D array): populated with 3 energy levels each, per 4 qubits
                      # n=0,n=1,n=2
                En =   [[[],[],[]], # Ej/Ec = 1
                        [[],[],[]], # Ej/Ec = 5
                        [[],[],[]], # Ej/Ec = 10
                        [[],[],[]]] # Ej/Ec = 50
            ng (1D array): np.linspace(-min, +max, #pts), default = np.linspace(-2.0, 2.0, 101)
        """
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        axs[0, 0].plot(ng, En[0][0], 'k')
        axs[0, 0].plot(ng, En[0][1], 'r')
        axs[0, 0].plot(ng, En[0][2], 'b')
        axs[0, 0].set_title(r'$a) \frac{E_j}{E_c} = 1.0$')
        axs[0, 0].set_xlabel(r'$n_g$')
        axs[0, 0].set_ylabel(r'$E_m / E_{01}$')


        axs[0, 1].plot(ng, En[1][0], 'k')
        axs[0, 1].plot(ng, En[1][1], 'r')
        axs[0, 1].plot(ng, En[1][2], 'b')
        axs[0, 1].set_title(r'b) $\frac{E_j}{E_c} = 5.0$')
        axs[0, 1].set_xlabel(r'$n_g$')
        axs[0, 1].set_ylabel(r'$E_m / E_{01}$')

        axs[1, 0].plot(ng, En[2][0], 'k')
        axs[1, 0].plot(ng, En[2][1], 'r')
        axs[1, 0].plot(ng, En[2][2], 'b')
        axs[1, 0].set_title(r'c) $\frac{E_j}{E_c} = 10.0$')
        axs[1, 0].set_xlabel(r'$n_g$')
        axs[1, 0].set_ylabel(r'$E_m / E_{01}$')

        axs[1, 1].plot(ng, En[3][0], 'k')
        axs[1, 1].plot(ng, En[3][1], 'r')
        axs[1, 1].plot(ng, En[3][2], 'b')
        axs[1, 1].set_title(r'd) $\frac{E_j}{E_c} = 50.0$')
        axs[1, 1].set_xlabel(r'$n_g$')
        axs[1, 1].set_ylabel(r'$E_m / E_{01}$')
