# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:29:04 2023

@author: User
"""

"""main_variational_method_NL_P03.py

Python script containing function definitions for minimizing the variational
integral for a user-specified potential function using a user-defined trial
function. As an example, the script implements the variational method for the
one-dimensional quantum harmonic oscillator using a Gaussian trial function.

Author: O. Melchert
Date: 2022-12-12
"""
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy.fft as nfft


def plot_results(x, Vx, E0_var, chi0_var, res_W):
    r"""plot results

    Generates a two-part figure, summarizing the results of the variational
    calculation.

    Args:
        x (array): discrete x-grid
        Vx (array): potential in array-form (not as a function)
        W0_var (float): minimum value of the variational integral
        chi0_var (array): normalized, approximate ground-state wavefunction
        res_W (array): sequence of energy values encountered during minimization

    Returns: nothing, but generates the figure "fig01.png" in the current
        working directory
    """
    # -- SET FIGURE DETAILS ---------------------------------------------------
    V_lim = (-2,2.)
    x_lim = (-3.5, 3.5)
    phi0 = 1./np.cosh(x)                                                            # C-P4
    # -- SET PLOT -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    plt.subplots_adjust(left=0.08, bottom=0.12, top=0.98, right=0.98)
    # ... LEFT SUBPLOT                                                               
    ax1.plot(range(1,res_W.size+1), res_W, color='k', marker='o', markersize=4, label=r'$W_\ell$')
    ax1.set_yscale('log')
    ax1.axhline(0, color='k', lw=0.5)
    ax1.set_ylabel("Variational integral $W$")
    ax1.set_xlabel("Iteration $\ell$")
    ax1.legend(frameon=False, loc=1)
    # ... RIGHT SUBPLOT
    ax2.plot(x, chi0_var, color="red", label=r"$\chi$")                             # C-P4
    ax2.fill_between(x, 0, phi0, color='C0', alpha=0.7, lw=0, label=r"$\phi_0$")    # C-P4
    ax2.plot(x, Vx, color="k", label=r"$V$")

    ax2.set_ylim(V_lim)
    ax2.tick_params(axis='y', direction='out', length=2, pad=1, right=False)
    ax2.set_ylabel("Trial function $\chi$")
    ax2.set_xlim(x_lim)
    ax2.tick_params(axis='x', direction='out', length=2, pad=1, top=False)
    ax2.set_xlabel("Position $x$")
    ax2.legend(frameon=False, loc=1)
    
    pos = ax1.get_position()
    fig.text(pos.x0, pos.y1,r"(a)" ,color='white',
            backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
            boxstyle='square,pad=0.1'), verticalalignment='top' )

    pos = ax2.get_position()
    fig.text(pos.x0, pos.y1,r"(b)" ,color='white',
            backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
            boxstyle='square,pad=0.1'), verticalalignment='top' )
    
    plt.savefig('figP03.png',dpi=600)
    #plt.show()


# def variational_method(x, mu, chi, a0):                                             # C-P1
#     r"""Variational method for calculating approximate ground-states.

#     Minimizes the variational integral via a user-defined trial function.

#     Notes:
#     -# The second order derivative in the first term is approximated using a
#         three-point finite difference stencil.
#     -# Works exculsively with non-normalized trial functions.

#     Args:
#         x (array): discrete x-grid
#         mu (float): propagation constant of self-localized solution
#         chi (function): parameterized trial function
#         a0 (float): adjustable parameter entering the trial function

#     Returns (alpha_star, chi0_var, W0_star, res_W):
#         alpha_star (float): optimal parameter value
#         chi0_var (array): self-localized solution (not normalized)
#         W0_var (float): minimum value of the variational integral
#         res_W (array): sequence of values encountered during minimization
#     """
#     dx = x[1]-x[0]

#     # -- STEP 1: MINIMIZE VARIATIONAL INTEGRAL --------------------------------
#     # ... DEFINE VARIATIONAL INTEGRAL
#     def W(alpha):
#         p = chi(alpha)
#         d2p_dx2 = (p[:-2] - 2*p[1:-1] + p[2:])/dx/dx
#         Vx = -p**2
#         return np.abs((np.trapz(-0.5*p[1:-1]*d2p_dx2, dx=dx)
#                 + np.trapz(p**2*Vx, dx=dx))/np.trapz(np.abs(p)**2,dx=dx) + mu)

#     # ... MINIMIZE VARIATIONAL INTEGRAL
#     res_W = []
#     cb_fun = lambda alpha: res_W.append(W(alpha))
#     res = so.minimize(W, a0, method='Nelder-Mead', tol=1e-8, callback = cb_fun)

#     # -- STEP 2: RETURN APPROXIMATE GROUND-STATE ------------------------------
#     return res.x, chi(res.x), W(res.x), np.asarray(res_W)

###############################################################################

def variational_method_spectral(x, mu, chi, a0):
    r"""Variational method for calculating approximate ground-states.

    Minimizes the variational integral via a user-defined trial function.

    Notes:
    -# The second order derivative is approximated using a second order spectral derivative.
    -# Works exculsively with non-normalized trial functions.

    Args:
        x (array): discrete x-grid
        mu (float): propagation constant of self-localized solution
        chi (function): parameterized trial function
        a0 (float): adjustable parameter entering the trial function

    Returns (alpha_star, chi0_var, W0_star, res_W):
        alpha_star (float): optimal parameter value
        chi0_var (array): self-localized solution (not normalized)
        W0_var (float): minimum value of the variational integral
        res_W (array): sequence of values encountered during minimization
    """
    dx = x[1]-x[0]
    N = len(x)
    k = nfft.fftfreq(N, d=dx)*2*np.pi

    # -- STEP 1: MINIMIZE VARIATIONAL INTEGRAL --------------------------------
    # ... DEFINE VARIATIONAL INTEGRAL
    def W(alpha):
        p = chi(alpha)
        
        # Compute the spectral derivative
        p_hat = nfft.ifft(p)     # Inverse Fourier Transform 
        d2p_dx2_hat = (-k**2) * p_hat #Second order soliton
        
        d2p_dx2 = nfft.fft(d2p_dx2_hat).real  # Fourier Transform
        
        Vx = -p**2
        return np.abs((np.trapz(-0.5*p*d2p_dx2, dx=dx)
                + np.trapz(p**2*Vx, dx=dx))/np.trapz(np.abs(p)**2,dx=dx) + mu)

    # ... MINIMIZE VARIATIONAL INTEGRAL
    res_W = []
    cb_fun = lambda alpha: res_W.append(W(alpha))
    res = so.minimize(W, a0, method='Nelder-Mead', tol=1e-12, callback = cb_fun)

    # -- STEP 2: RETURN APPROXIMATE GROUND-STATE ------------------------------
    return res.x, chi(res.x), W(res.x), np.asarray(res_W)


def main():
    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS ---------------------
    x = np.linspace(-30, 30, 1000, endpoint=True)
    #print(x)
    mu = 0.5
    chi_A = lambda alpha: np.where(np.abs(x)<alpha, (1-(x/alpha)**2)**8, 0)
    chi_B = lambda alpha: alpha/np.cosh(x*alpha)
    chi_C = lambda alpha: alpha[0]*np.exp(-x**2/(2*alpha[1]**2))
    chi = chi_C
    alpha = [1,1] #adjustable parameters
    
    #for alpha in alphas:
    # -- (2) PERFORM COMPUTATION - OBTRAIN GROUND STATE VIA VARIATION METHOD --
    alpha_star, chi0_var, W0_var, res_W = variational_method_spectral(x, mu, chi, alpha)
    #alpha_star, chi0_var, W0_var, res_W = variational_method(x, mu, chi, alpha)
    
    print("alpha_star =", alpha_star) #Optimized parameter
    print("W(alpha_star) =", W0_var)  #Minimum value of variational integral
    
    #print("(chi0_var) =\n", chi0_var) #Approximate ground-state wavefunction

    # -- (3) POSTPROCESS RESULTS - GENERATE FIGURE ---------------------------- 
    print("# E =",np.trapz(np.abs(chi0_var)**2,x=x))
    plot_results(x, -np.abs(chi0_var)**2, W0_var, chi0_var, res_W)

if __name__=="__main__":
    main()
