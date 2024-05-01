import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time


def pack_pars(Tdeg, cat_conc, cocat_conc):
    """
    Function which packs parameters and initial conditions into a list

    :param Tdeg: reaction temperature, in °C
    :param cat_conc: catalyst concentration, in m3/mol
    :param cocat_conc: cocatalyst concentration, in m3/mol
    :return: returns a list of parameters used in the model equations
    """
    # General physical-chemical constants
    R = 8.314  # Universal gas constant, J/K/mol
    N_A = 6.022140858e23  # Avogadro number, 1/mol
    MW = 72.06e-3  # Molecular weight of lactoyl (monomer) group, kg/mol

    # Reaction temperature, deg. C
    T = Tdeg + 273.15  # Reaction temperature, K
    M0 = 1.0e4  # Init. conc. of monomer, mol/m3
    C0 = cat_conc       #M0 / 1.0e4  # Init. conc. of catalyst, mol/m3
    D0 = cocat_conc     #C0 * 60.0  # Init. conc. of cocatalyst, mol/m3
    A0 = C0 * 0.36  # Init. conc. of acid, mol/m3

    # KINETIC PARAMETERS
    # Activation & Deactivation (for all temperatures)
    k_a1 = 1.0e6 * 1.0e-3 / 3.6e3  # Activation rate coefficient, m3/mol/s
    Keq_a = 0.256  # Activation equilibrium constant, -
    k_a2 = k_a1 / Keq_a  # Deactivation rate coefficient, m3/mol/s

    # Propagation & Depropagation
    k_p0 = 7.4e11 * 1.0e-3 / 3.6e3  # Preexponential factor, m3/mol/s
    Ea_p = 63.3 * 1e3  # Activation energy, J/mol
    k_p = k_p0 * np.exp(-Ea_p / (R * T))  # Propagation rate coefficient, m3/mol/s
    Meq = 0.225 * 1e3  # Monomer equilibirum constant, mol/m3
    k_d = k_p * Meq  # Depropagation rate coefficient, 1/s

    # Chain transfer (for all temperatures)
    k_s = 1.0e6 * 1.0e-3 / 3.6e3  # Rate coefficient, m3/mol/s

    # Intermolecular transesterification
    k_te0 = 3.38e11 * 1.0e-3 / 3.6e3  # Preexponential factor, m3/mol/s
    Ea_te = 83.3e3  # Activation energy, J/mol
    k_te = k_te0 * np.exp(-Ea_te / (R * T))  # Rate coefficient, m3/mol/s

    # Nonradical random chain scission
    k_de0 = 1.69e8 / 3.6e3  # Preexponential factor, 1/s
    Ea_de = 101.5e3  # Activation energy, J/mol
    k_de = k_de0 * (np.exp(-Ea_de / (R * T)))

    mu00 = D0
    la00 = 1e-7
    la10 = 1e-7
    la20 = 1e-7
    mu10 = 1e-7
    mu20 = 1e-7
    ga00 = 1e-7
    ga10 = 1e-7
    ga20 = 1e-7
    pars = [k_a1, k_a2, k_s, k_te, k_de, k_d, k_p]
    y0 = [M0, C0, A0, la00, la10, la20, mu00, mu10, mu20, ga00, ga10, ga20]
    return pars, y0


# Moment closure equations
# la3 = la2 * (2.0 * la2 * la0 - la1 ** 2) / (la1 * la0)
# mu3 = mu2 * (2.0 * mu2 * mu0 - mu1 ** 2) / (mu1 * mu0)
# ga3 = ga2 * (2.0 * ga2 * ga0 - ga1 ** 2) / (ga1 * ga0)

dydt = [0]*12
# System of ordinary differential equations
def ODE_equations(y, t, pars):
    # unpack parameters
    #dydt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k_a1, k_a2, k_s, k_te, k_de, k_d, k_p = pars

    M, C, A, la0, la1, la2, mu0, mu1, mu2, ga0, ga1, ga2 = y

    la3 = la2 * (2.0 * la2 * la0 - la1 ** 2) / (la1 * la0)
    mu3 = mu2 * (2.0 * mu2 * mu0 - mu1 ** 2) / (mu1 * mu0)
    ga3 = ga2 * (2.0 * ga2 * ga0 - ga1 ** 2) / (ga1 * ga0)

    # Balance of monomer, lactide

    dydt[0] = -k_p * M * la0 + k_d * la0

    # Balance of catalyst, tin(II) octoate
    dydt[1] = -k_a1 * C * mu0 + k_a2 * A * la0

    # Balance of octanoic acid
    dydt[2] = k_a1 * C * mu0 - k_a2 * A * la0

    # Balance of 0th moment (concentration) of active chains
    dydt[3] = k_a1 * mu0 * C - k_a2 * la0 * A

    # Balance of 1st moment of active chains
    dydt[4] = k_a1 * mu1 * C - k_a2 * la1 * A + 2.0 * k_p * M * la0 - 2.0 * k_d * la0 - k_s * la1 * mu0 \
              + k_s * mu1 * la0 - k_te * la1 * (mu1 - mu0) + 0.5 * k_te * la0 * (mu2 - mu1) - k_te * la1 * (ga1 - ga0) \
              + 0.5 * k_te * la0 * (ga2 - ga1) - 0.5 * k_de * (la2 - la1)

    # Balance of 2nd moment of active chains
    dydt[5] = k_a1 * mu2 * C - k_a2 * la2 * A + 4.0 * k_p * M * (la1 + la0) + 4.0 * k_d * (la0 - la1) - k_s * la2 * mu0 \
              + k_s * mu2 * la0 + (1.0 / 3.0) * k_te * la0 * (la1 - la3) + k_te * la1 * (la2 - la1) \
              - k_te * la2 * (mu1 - mu0) + (1.0 / 6.0) * k_te * la0 * (2.0 * mu3 - 3.0 * mu2 + mu1) \
              - k_te * la2 * (ga1 - ga0) + (1.0 / 6.0) * k_te * la0 * (2.0 * ga3 - 3.0 * ga2 + ga1) \
              - (1.0 / 6.0) * k_de * (4.0 * la3 - 3.0 * la2 - la1)

    # Balance of 0th moment (concentration) of dormant (all OH-bearing) species
    dydt[6] = -k_a1 * mu0 * C + k_a2 * la0 * A

    # Balance of 1st moment of dormant chains
    dydt[7] = -k_a1 * mu1 * C + k_a2 * la1 * A + k_s * la1 * mu0 - k_s * mu1 * la0 + k_te * la1 * (mu1 - mu0) \
              - (1.0 / 2.0) * k_te * la0 * (mu2 - mu1) - (1.0 / 2.0) * k_de * (mu2 - mu1)

    # Balance of 2nd moment of dormant chains
    dydt[8] = -k_a1 * mu2 * C + k_a2 * la2 * A + k_s * la2 * mu0 - k_s * mu2 * la0 + k_te * la2 * (mu1 - mu0) \
              + k_te * la1 * (mu2 - mu1) + (1.0 / 6.0) * k_te * la0 * (-4.0 * mu3 + 3.0 * mu2 + mu1) \
              - (1.0 / 6.0) * k_de * (4.0 * mu3 - 3.0 * mu2 - mu1)

    # Balance of 0th moment (concentration) of terminated chains
    dydt[9] = k_de * (la1 - la0) + k_de * (mu1 - mu0)

    # Balance of 1st moment of terminated chains
    dydt[10] = k_te * la1 * (ga1 - ga0) - 0.5 * k_te * la0 * (ga2 - ga1) - k_de * (ga2 - ga1) \
               + 0.5 * k_de * (la2 - la1) + 0.5 * k_de * (mu2 - mu1)

    # Balance of 2nd moment of terminated chains
    dydt[11] = k_te * la2 * (ga1 - ga0) + k_te * la1 * (ga2 - ga1) + (1.0 / 6.0) * k_te * la0 \
               * (-4.0 * ga3 + 3.0 * ga2 + ga1) - (1.0 / 3.0) * k_de * (4.0 * ga3 - 3.0 * ga2 - ga1) + (1.0 / 6.0) \
               * k_de * (2.0 * la3 - 3.0 * la2 + la1) + (1.0 / 6.0) * k_de * (2.0 * mu3 - 3.0 * mu2 + mu1)

    return dydt



def main_func(Tdeg, cat_conc, cocat_conc):
    pars,y0 = pack_pars(Tdeg, cat_conc, cocat_conc)
    t = np.arange(0, 0.2 * 3600, 0.1)
    #TODO prepsat ode_solutions[neco,neco] pomocí la0,la1 atd
    ode_solution = odeint(ODE_equations, y0=y0, t=t,args=tuple([pars]),mxstep=1000)
    nulty_moment = ode_solution[-1, 3] +ode_solution[-1, 6]+ode_solution[-1, 9]
    prvni_moment = ode_solution[-1, 4] +ode_solution[-1, 7]+ode_solution[-1, 10]
    druhy_moment = ode_solution[-1, 5] +ode_solution[-1, 8]+ode_solution[-1, 11]

    MW = 72.06e-3  # Molecular weight of lactoyl
    uw = (ode_solution[:, 4] +ode_solution[:, 7]+ode_solution[:, 10])/(ode_solution[:, 3] +ode_solution[:, 6]+ode_solution[:, 9])*MW
    un = (ode_solution[:, 5] +ode_solution[:, 8]+ode_solution[:, 11])/(ode_solution[:, 4] +ode_solution[:, 7]+ode_solution[:, 10])*MW
    stredni_pocetni_delka_retezce = (prvni_moment/nulty_moment)*MW
    stredni_hmotnostni_delka_retezce = (druhy_moment/prvni_moment)*MW
    M0 = y0[0]
    M = ode_solution[:,0]
    konverze  = (M0-M[-1])/M0*100
    # plt.plot(t, ode_solution[:, 0], label="Monomer concentration")
    # plt.plot(t, ode_solution[:, 1], label="Catalyst concentration")
    # plt.plot(t, ode_solution[:, 2], label="Acid concentration")
    # plt.plot(t, ode_solution[:, 3], label="0th moment of active chains")
    # plt.plot(t, ode_solution[:, 4], label="1st moment of active chains")
    # #plt.plot(t, ode_solution[:, 5], label="2nd moment of active chains")
    # plt.plot(t, ode_solution[:, 6], label="0th moment of dormant chains")
    # #plt.plot(t, ode_solution[:, 7], label="1st moment of dormant chains")
    # #plt.plot(t, ode_solution[:, 8], label="2nd moment of dormant chains")
    # plt.plot(t, ode_solution[:, 9], label="0th moment of terminated chains")
    # plt.plot(t, ode_solution[:, 10], label="1st moment of terminated chains")
    # #plt.plot(t, ode_solution[:, 11], label="2nd moment of terminated chains")
    #
    # plt.suptitle("Solved system of ODEs", fontsize=14)
    # plt.legend()
    # plt.xlabel("time [s]", fontsize=10)
    # plt.ylabel("concentration mol/m3", fontsize=10)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.plot(t,uw)
    # plt.plot(t,un)
    # plt.show()

    return [stredni_pocetni_delka_retezce, stredni_hmotnostni_delka_retezce,konverze]
"""
inputs = teplota, koncentrace kat a kokat
outputs = stredni pocetni delka retezce (suma momentu prvniho radu / sum mom 0 radu)
outputs = stredni hmotnostni delka retezce (suma momentu druheho radu / sum mom 1 radu)

CLD = chain length distribution

weight average and number average of CLD

weight average of CLD * molecular weight of monomer = weight average molecular weight

12 slidu, vysvetlit podstatu jak se uci NN, najit cas pro ustaleny stav (kdyztak zafixovat pro 0.2 hod)
"""

#main_func(140,2,90)