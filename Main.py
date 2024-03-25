# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:52:10 2022

@author: Eugenio
"""

#%% Standard Modules

import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#%% Internal Modules

from Functions.supporting_functions import stdBm_loop, stdBm_matrix, stdBm_bridge, convergenceTestBm

#%% Process

def run(latex_mode, quasiMC, nPathsPower, nSteps, horizon, x0, methods):
    
    if (latex_mode):
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsfonts}'
             })
        
    dfExp, dfVar = convergenceTestBm(quasiMC, nPathsPower, nSteps, horizon, x0, methods)
           
    return dfExp, dfVar
    

#%% Simulate Standard Brownian

if __name__ == "__main__":
    
    # Params ------------------------------------------------------------------    
    quasiMC     = True
    latex_mode  = True
    methods     = ["loop", "matrix", "bridge"]    
    nPathsPower = [2, 3, 4, 5, 6, 7, 8, 9, 10] #nPaths = 2**nPathsPower    
    nSteps      = 2**10 #should be power two for Bridge and Sobol 
    horizon     = 1.0
    x0          = 0.0
    nPaths      = 3
    
    # Convergence -------------------------------------------------------------
    dfExp, dfVar = run(latex_mode, quasiMC, nPathsPower, nSteps, horizon, x0, methods)
    
    # Visualize Convergence ---------------------------------------------------
    plt.figure(1)
    plt.plot(2**dfExp.index, dfExp, label = "Expectation", marker = "o")
    plt.title("Expectation")
    plt.legend(methods)
    plt.xlabel(xlabel="N paths")    
    
    plt.figure(2)
    plt.plot(2**dfVar.index, dfVar, label = "Variance", marker = "o")
    plt.title("Variance")
    plt.legend(methods)
    plt.xlabel(xlabel="N paths")
    
    # Sim ---------------------------------------------------------------------
    sim_loop   = stdBm_loop(nPaths, nSteps, horizon, x0)    
    sim_mat    = stdBm_matrix(nPaths, nSteps, horizon, x0) 
    sim_bridge = stdBm_bridge(nPaths, nSteps, horizon, x0, quasiMC=True) 
        
    # Visualize Sim -----------------------------------------------------------    
    plt.figure(3)
    plt.plot(sim_loop["time"], sim_loop["W"].T, label = "loop" )
    plt.title(f"Loop: N paths {2**nPaths}")
    
    plt.figure(4)
    plt.plot(sim_mat["time"], sim_mat["W"], label = "matrix")
    plt.title(f"Matrix: N paths {2**nPaths}")
    
    plt.figure(5)
    plt.plot(sim_bridge["time"], sim_bridge["W"].T, label = "bridge")
    plt.title(f"Bridge: N paths {2**nPaths}")
