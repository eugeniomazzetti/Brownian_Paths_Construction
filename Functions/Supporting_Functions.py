# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:52:53 2022

@author: Eugenio
"""

#%% Standard Modules

import numpy as np
import scipy.stats.qmc as qmc
import scipy.stats as sts
import pandas as pd

#%% Brownian Motion Sim: Incremental

def stdBm_incremental(powerTwoPaths:int, nSteps:int, T:int, x0:float=0.0) -> dict:
    """
    Incremental path construction:
        
        W(t_{i+1}) - W(t_{i}) =d Z*sqrt(t_{i+1}-t_{i})

    with =d meaning convergence in distribution. Therefore:
        
        W(t_{i+1}) = W(t_{i}) + Z*sqrt(t_{i+1}-t_{i})
    
    Parameters
    ----------
    powerTwoPaths : int
        Number of paths is equal to 2^powerTwoPaths.
    nSteps : int
        Number of steps.
    T : int
        Final simulation time.
    x0 : float
        Initial value. For a standard Bm it should be zero. Default is 0.0.

    Returns
    -------
    output : dict
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    
    #Initialize    
    Z      = np.random.normal(0.0, 1.0, [2**powerTwoPaths, nSteps])
    W      = np.zeros([2**powerTwoPaths, nSteps + 1])
    time   = np.zeros([nSteps + 1])    
    W[:,0] = x0
    dt     = T/nSteps
    
    #Loop
    for i in range(0, nSteps):
        # NOTE: Centering improves drastically convergence of this method.
        # if 2**powerTwoPaths > 2:
        #     Z[:,i] = (Z[:,i] -np.mean(Z[:,i]))/np.std(Z[:,i])
        
        W[:,i+1]  = W[:,i] + Z[:,i]*np.sqrt(dt)
        time[i+1] = time[i] + dt
        
    #Output    
    output = {"time":time, "W":W}
    
    return output
        
#%% Brownian Motion Sim: Matrix

def stdBm_cholesky(powerTwoPaths:int, nSteps:int, T:int, x0:float=0.0) -> dict:
    """
    Simulates brownian paths using linear algebra. It is equivalent to the 
    incremental path construction. From linear algebra we know that any 
    positive-definite matrix can be decomposed as:
        
                                C = AA'
    
    We use Cholesky decomposition to find the square-root of the symmetric positive-definite 
    AUTO-covariance matrix of a Brownian path.
    
    Note that the AUTO-covaraince matrix of a Brownian path is the AUTO-covariance 
    matrix of Brownian motions (random) vector; e.g. [W(t1), W(t2), W(t3)].    
    Given that Cov(W(t), W(s)) = E[W(t)W(s)] = min(t,s), the AUTO-covariance 
    matrix and its Cholesky are:
        
      AUTO-covariance                Cholesky                     N(0,1)
      ---------------      ------------------------------------   ------   
      | t1  t1   t1 |      |sqrt(t1)      0            0      |   | Z1 |
      | t1  t2   t2 | -->  |sqrt(t1) sqrt(t2-t1)       0      | X | Z2 | 
      | t1  t2   t3 |      |sqrt(t1) sqrt(t2-t1)  sqrt(t3-t2) |   | Z3 |
      ---------------      ------------------------------------   ------
      
    Therefore a Brownian path can be constructed multiplying Cholesky times a vector
    Z of standard normal r.v. . This is equivalent to the incremental path construction.
    
    NOTE: The decomposition of C via Cholesky is not the only available one. See PCA
          and Spectral methods.
    
    Parameters
    ----------
    powerTwoPaths : int
        Number of paths is equal to 2^powerTwoPaths.
    nSteps : int
        Number of steps.
    T : int
        Final simulation time.
    x0 : float
        Initial value. For a standard Bm it should be zero. Default is 0.0.

    Returns
    -------
    output : dict
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    
    #Initalize
    dt       = T/nSteps
    matrix   = np.full([nSteps, nSteps], np.sqrt(dt))
    cholesky = np.tril(matrix)
    Z        = np.random.normal(0.0, 1.0, [2**powerTwoPaths, nSteps]) 
    
    #Simulation
    X          = np.matmul(cholesky, np.transpose(Z))
    X0_vec     = np.zeros([2**powerTwoPaths]) + x0
    X_final    = np.vstack([X0_vec, x0 + X])      
    time       = np.hstack((0.0, np.cumsum(np.full([1, nSteps],dt))))
    
    #Output
    output     = {"time":time, "W":X_final}
    
    return output

#%% Brownian Motion Sim: PCA

def stdBm_pca(powerTwoPaths:int, nSteps:int, nFactors:int, T:int, x0:float=0.0)->dict:
    """
    From linear algebra we know that any positive-definite matrix can be decomposed
    as:
        
                                     C = AA'
    
    We use Eigen-Decomposition to find the square-root of the symmetric positive-definite 
    AUTO-covariance matrix of a Brownian path. This permits to select nFactors
    or Principal Components < nSteps explaining most of the path variability. 
    
    Note that the AUTO-covaraince matrix of a Brownian path is the AUTO-covariance 
    matrix of Brownian motions (random) vector; e.g. [W(t1), W(t2), W(t3)].        
    Given that Cov(W(t), W(s)) = E[W(t)W(s)] = min(t,s), the AUTO-covariance 
    matrix and its Cholesky are:
        
      AUTO-covariance              O                     L                N(0,1)
      ---------------      ----------------   ------------------------    -----
      | t1  t1   t1 |      | e11 e12  e13 |   |sqrt(l1)  0      0    |   | Z1 |
      | t1  t2   t2 | -->  | e21 e22  e23 | X |   0  sqrt(l2)   0    | X | Z2 | 
      | t1  t2   t3 |      | e31 e32  e33 |   |   0      0   sqrt(l3)|   | Z3 |
      ---------------      ----------------   ------------------------   -----
    
    Indeed the Eigen-Decomposition of a square symmetric matrix is:
        
                  C = OLO' = O sqrt(L)*[O sqrt(L)]' = AA'  
    
    Selecting the first n columns of O and n eigenvalues will effectively reduce
    the dimension of the simulation.
    
    Parameters
    ----------
    powerTwoPaths : int
        Number of paths is equal to 2^powerTwoPaths.
    nSteps : int
        Number of steps.
    nFactors : int
        Using PCA permits the selection of M <= N = nsteps most important factors.    
    T : int
        Final simulation time.
    x0 : float
        Initial value. For a standard Bm it should be zero. Default is 0.0.
    
    Returns
    -------
    output : dict
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    if nSteps < nFactors:
        raise Exception("nFactors <= nSteps")
    
    #Initalize
    dt      = T/nSteps
    st      = np.mgrid[1:(nSteps+1), 1:(nSteps+1)]
    autoCov = dt*st.min(axis=0) 
    
    #Diagonalize: PCA
    eigenVal, eigenVec = np.linalg.eigh(autoCov) #Symmetric Matrix       
    idx      = eigenVal.argsort()[::-1]   
    eigenVal = eigenVal[idx]
    eigenVec = eigenVec[:, idx]    
    sqrtMat  = eigenVec[:, :nFactors] * np.sqrt(eigenVal[:nFactors])        
    Z        = np.random.normal(0.0, 1.0, [2**powerTwoPaths, nFactors]) 
   
    #Simulation
    X          = np.matmul(sqrtMat, np.transpose(Z))
    X0_vec     = np.zeros([2**powerTwoPaths]) + x0
    X_final    = np.vstack([X0_vec, x0 + X])      
    time       = np.hstack((0.0, np.cumsum(np.full([1, nSteps],dt))))
        
    #Output
    output = {"time":time, "W":X_final}
    
    return output

#%% Brownian Motion Sim: Spectral

def stdBm_spectral(powerTwoPaths:int, nSteps:int, T:int, x0:float=0.0) -> dict:
    
    """
    From linear algebra we know that any positive-definite matrix can be decomposed
    as:
        
                                C = AA'
    
    For EQUIDISTANT step-size of length "dt" the AUTO-covariance matrix of a Brownian 
    path has the form:
        
                               AUTO-covariance   
                            ---------------------   
                            | 1  1   1   1 ... 1|   
                            | 1  2   2   2 ... 2| 
                            | 1  2   3   3 ... 3| 
                      dt *  | .  .   .   . .   .|
                            | .  .   .   .   . .| 
                            | .  .   .   .     .|                             
                            | 1  2   3   3 ... N| 
                            ---------------------   
   
    It can be proved that the elements of the square-root amtrix A are then:

    A[k,l] = sqrt(dt)/sqrt(2N+1)* sin(k*pi*(2l-1)/(2n+1)) / sin(0.5*pi*(2l-1)/(2n+1))                             
      
    
    Parameters
    ----------
    powerTwoPaths : int
        Number of paths is equal to 2^powerTwoPaths.
    nSteps : int
        Number of steps.
    T : int
        Final simulation time.
    x0 : float
        Initial value. For a standard Bm it should be zero. Default is 0.0.

    Returns
    -------
    output : dict
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    
    dt      = T/nSteps
    denom   = 2*nSteps + 1
    const   = np.sqrt(dt/denom)
    sqrtMat = np.zeros([nSteps, nSteps])
    
    for k in range(0,nSteps):
        for l in range(0,nSteps):
            num         = 2*(l+1)-1
            factor      = num/denom
            sqrtMat[k,l]= const*np.sin((k+1)*np.pi*factor)/np.sin(0.5*np.pi*factor)
        
    #Simulation
    Z       = np.random.normal(0.0, 1.0, [2**powerTwoPaths, nSteps])    
    X       = np.matmul(sqrtMat, np.transpose(Z))
    X0_vec  = np.zeros([2**powerTwoPaths]) + x0
    X_final = np.vstack([X0_vec, x0 + X])      
    time    = np.hstack((0.0, np.cumsum(np.full([1, nSteps],dt))))
        
    #Output
    output = {"time":time, "W":X_final}
    
    return output

#%% Brownian Motion Sim: Brownian Bridge 

def stdBm_bridge(powerTwoPaths:int, nSteps:int, T:int, x0:float=0.0, quasiMC:bool=True)->dict:
    """
    The function simulates brownian paths via brownian bridge construction. 
    See Glasserman p.85.
    
    The idea is to have T = 2^m time-steps therefore a time-vector of length m+1
    starting at zero:
    
        (i)   Divide the interval in two segments  l = 0 < i = 2^m/2 < r = 2^m.
        (ii)  Knowing W(l=0) = x0 and W(r=T) = Z(T)*sqrt(T) sample W(i = T/2) using
              a Brownian bridge with endpoints [W(l=0), W(r = T)].
        (iii) repeat m times populating each time the 2^k segments k = 1,2,...,m 
    
    The function supports the use of Sobol low-discrepancy numbers. Notes that
    in this setting (quasiMC = True):
        
        nSteps = Sobol dimension
        nPaths = Sobol points
    
    Parameters
    ----------
    powerTwoPaths : int
        Number of paths is 2**powerTwoPaths.
    nSteps : int
        Number of steps.
    T : int
        Final simulation time.
    x0 : float
        Initial value. For a standard Bm it should be zero. Default is 0.0.
    quasiMC : boolean    
        If True, perform Quasi MC simulation via Sobol sequence as opposed to 
        usual MC backed by pseudo random numbers. The default is True.
    Returns
    -------
    output : dict
        Contains (i)  time: vector of time points
                 (ii) W   : Brownian simulation paths

    """
    #Initialize
    dt     = T/nSteps
    time   = np.cumsum(np.full([nSteps],dt))         
    time   = np.hstack((0.0, time))
    if (quasiMC):
        Z  = sts.norm.ppf(qmc.Sobol(d = nSteps, scramble=True).random_base2(m=powerTwoPaths))    
    else:         
        Z  = np.random.normal(0.0, 1.0, [2**powerTwoPaths, nSteps]) 
    h      = nSteps
    j_max  = 1
    W   = np.zeros([2**powerTwoPaths, nSteps+1])        
    W_h     = Z[:, -1]*np.sqrt(time[-1])   
    W[:,0]  = x0
    W[:,-1] = W_h
    m       = np.log(nSteps)/np.log(2)
   
    for k in range(0, int(m)):   
       
        i_min = int(h/2)
        i     = int(i_min)
        l     = 0
        r     = int(h)
       
        for j in range(1, j_max + 1):
           
            a = ((time[r]-time[i])*W[:,l] + (time[i]-time[l])*W[:,r])/(time[r]-time[l]) #Expectation
            b = np.sqrt((time[i]-time[l])*(time[r]-time[i])/(time[r]-time[l]))          #Std Dev.
           
            W[:,i] = a + b*Z[:,i]
            i = int(i + h)
            l = int(l + h)
            r = int(r + h)
           
        j_max = 2*j
        h     = i_min
      
    output = {"time":time, "W":W}
    return output   

#%% Fractional Brownian Motion Sim

#TODO: Implement circulant matrix approach and others.
        
#%% Convergence Test

def convergenceTestBm(quasiMC, nPathsPower, nSteps, nFactors, horizon, x0, methods):
    
    dfExp = pd.DataFrame(columns=methods)
    dfVar = pd.DataFrame(columns=methods)
    
    for i, exp in enumerate(nPathsPower):
        sim_loop   = stdBm_incremental(exp, nSteps, horizon, x0)        
        sim_mat    = stdBm_cholesky(exp, nSteps, horizon, x0)         
        sim_bridge = stdBm_bridge(exp, nSteps, horizon, x0, quasiMC) 
        sim_pca    = stdBm_pca(exp, nSteps, nFactors,T=horizon)   
        sim_spec   = stdBm_spectral(exp, nSteps, horizon, x0)
        
        
        dfExp.loc[i] = [np.mean(sim_loop["W"][:,-1]), 
                        np.mean(sim_mat["W"][-1,:]), 
                        np.mean(sim_bridge["W"][:,-1]),
                        np.mean(sim_pca["W"][-1,:]),
                        np.mean(sim_spec["W"][-1,:])]      
        dfVar.loc[i] = [(np.var(sim_loop["W"][:,-1])-horizon)/horizon, 
                        (np.var(sim_mat["W"][-1,:])-horizon)/horizon,
                        (np.var(sim_bridge["W"][:,-1])-horizon)/horizon,
                        (np.var(sim_pca["W"][-1,:])-horizon)/horizon,
                        (np.var(sim_spec["W"][-1,:])-horizon)/horizon]
    
    dfExp.index = nPathsPower
    dfVar.index = nPathsPower
    
    return dfExp, dfVar
    
