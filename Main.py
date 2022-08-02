# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:52:10 2022

@author: Eugenio
"""

#%% Import Standard Modules

import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")

#%% Import Internal Modules

from Functions.Supporting_Functions import std_bm_engine_loop, std_bm_engine_matrix, std_bm_engine_bridge

#%% Simulate STD Brownian: For Loop Vs. Matrix Vs. Bridge

# Params ----------------------------------------------------------------------

n_paths = 2
n_steps = 2**10
T       = 2
X0      = 0

# Sim ------------------------------------------------------------------------- 

sim_loop   = std_bm_engine_loop(n_paths, n_steps, T, X0)
sim_mat    = std_bm_engine_matrix(n_paths, n_steps, T, X0) 
sim_bridge = std_bm_engine_bridge(n_paths, n_steps, T, X0) 

print("Mean for loop:",np.mean(sim_loop["W"][:,-1]))
print("Variance for loop:", np.var(sim_loop["W"][:,-1]))

print("Mean Matrix:", np.mean(sim_mat["W"][-1,:]))
print("Variance Matrix:", np.var(sim_mat["W"][-1,:]))

print("Mean Bridge:", np.mean(sim_bridge["W"][-1,:]))
print("Variance Bridge:", np.var(sim_bridge["W"][-1,:]))


#%% Visualize

plt.figure(0)
plt.plot(sim_loop["time"], sim_loop["W"].T, label = "loop" )
plt.title("Loop")

plt.figure(1)
plt.plot(sim_mat["time"], sim_mat["W"], label = "matrix")
plt.title("Matrix")

plt.figure(2)
plt.plot(sim_bridge["time"], sim_bridge["W"], label = "bridge")
plt.title("Bridge")

