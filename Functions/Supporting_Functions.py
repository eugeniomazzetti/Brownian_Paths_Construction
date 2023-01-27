# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:52:53 2022

@author: Eugenio
"""

#%% Standard Modules

import numpy as np

#%% Brownian Motion Simulation: Incremental

def std_bm_engine_loop(n_paths, n_steps, T, X0):
    """
    Parameters
    ----------
    n_paths : int
        Number of paths.
    n_steps : int
        Number of steps.
    T : int
        Final simulation time.
    X0 : float
        Initial value.

    Returns
    -------
    output : dictionary
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    
    #Initialize    
    Z    = np.random.normal(0.0, 1.0, [n_paths, n_steps])
    W    = np.zeros([n_paths, n_steps +1])
    time = np.zeros([n_steps + 1])
    
    W[:,0] = X0
    dt     = T/int(n_steps)
    
    #Loop
    for i in range(0, n_steps):
        if n_paths > 1:
            Z[:,i] = (Z[:,i] -np.mean(Z[:,i]))/np.std(Z[:,i])
        
        W[:,i+1]  = W[:,i] + Z[:,i]*np.sqrt(dt)
        time[i+1] = time[i] + dt
        
    #Output    
    output = {"time": time, "W": W}
    
    return output
        
#%% Brownian Motion Simulation: Matrix

def std_bm_engine_matrix(n_paths, n_steps, T, X0):
    """
    This function simulates brownian paths using matrix algebra. 
    
    Note in fact that the variance-covaraince matrix of a Brownian path can be 
    seen as the varince-covariance matrix of a vector of Brownian motions. e.g. 
    [W(t1), W(t2), W(t3)].
    
    The variance-covariance matrix and its Cholesky look like:
        
    variance-covariance                Cholesky                   N(0,1)
      ---------------      ------------------------------------   ------   
      | t1  t1   t1 |      |sqrt(t1)      0            0      |   | Z1 |
      | t1  t2   t2 | -->  |sqrt(t1) sqrt(t2-t1)       0      | X | Z2 | 
      | t1  t2   t3 |      |sqrt(t1) sqrt(t2-t1)  sqrt(t3-t2) |   | Z3 |
      ---------------      ------------------------------------   ------
      
    Therefore Brownian path can be constructed multiplying Cholesky times a vector
    Z of standard normal r.v.
      
    Parameters
    ----------
    n_paths : int
        Number of paths.
    n_steps : int
        Number of steps.
    T : int
        Final simulation time.
    X0 : float
        Initial value.

    Returns
    -------
    output : dictionary
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    
    #Initalize
    dt         = T/int(n_steps)
    matrix     = np.full([n_steps, n_steps], np.sqrt(dt))
    cholesky   = np.tril(matrix)
    Z          = np.random.normal(0.0, 1.0, [n_steps, n_paths]) 
    
    #Simulation
    X          = np.matmul(cholesky, Z)
    X0_vec     = np.zeros([n_paths]) + X0
    X_final    = np.vstack([X0_vec, X0 + X])      
    time       = np.hstack((0.0,np.cumsum(np.full([1, n_steps],dt))))
    
    #Output
    output     = {"time": time, "W": X_final}
    
    return output
    
#%% Brownian Motion Simulation: Bridge Construction

def std_bm_engine_bridge(n_paths,n_steps, T, X0):
    """
    The function simulates brownian paths via brownian bridge construction. 
    See Glasserman p.85.
    
    The idea is to have T = 2^m time-steps therefore a time-vector of length m+1
    starting at zero.
    
    (i)   Divide the interval in two segments  l = 0 < i = 2^m/2 < r = 2^m.
    (ii)  Knowing W(l=0) = X0 and W(r=T) = Z(T)*sqrt(T) sample W(i = T/2) using
          a Brownian bridge with endpoints [W(l=0), W(r = T)].
    (iii) repeat m times populating each time the 2^k segments k = 1,2,...,m    
    
    Parameters
    ----------
    n_paths : int
        Number of paths.
    n_steps : int
        Number of steps.
    T : int
        Final simulation time.
    X0 : float
        Initial value.

    Returns
    -------
    output : dictionary
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    #Initialize
    dt    = T/n_steps
    time  = np.cumsum(np.full([n_steps],dt))         
    time  = np.hstack((0.0,time))
         
    Z      = np.random.normal(0.0, 1.0, [n_steps, n_paths]) 
    h      = n_steps
    j_max  = 1
   
    W       = np.zeros([n_steps+1, n_paths])
    W_h     = Z[-1, :]*np.sqrt(time[-1])   
    W[0,:]  = X0
    W[-1,:] = W_h
   
    for k in range(0, int(np.log(n_steps)/np.log(2))):   
       
        i_min = int(h/2)
        i     = int(i_min)
        l     = 0
        r     = int(h)
       
        for j in range(1, j_max + 1):
           
            a = ((time[r]-time[i])*W[l,:] + (time[i]-time[l])*W[r,:])/(time[r]-time[l])
            b = np.sqrt((time[i]-time[l])*(time[r]-time[i])/(time[r]-time[l]))
           
            W[i,:] = a + b*Z[i,:]
            i = int(i + h)
            l = int(l + h)
            r = int(r + h)
           
        j_max = 2*j
        h     = i_min
      
    output = {"time": time, "W":W}
    return output   
        
