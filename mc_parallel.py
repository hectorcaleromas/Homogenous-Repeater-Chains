import main
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from pathlib import Path
import pickle
from policy import Agent
import random
import time
import copy
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

## Simulation Parameters
n=5
p=1
p_s=1
cutoff=100
policy="efms-nested"
N_samples=1000
randomseed=2
seed_a=main.seed_array(randomseed,N_samples)
progress_bar='plain'
savedata=False
num_nodes=7
max_distance=2

## MC iteration definition


if __name__ == '__main__':
    if progress_bar == 'notebook':
        tqdm_ = tqdm_notebook
    elif progress_bar == 'plain':
        tqdm_ = tqdm
    else:
        raise ValueError('Invalid value for progress_bar.')

    if policy == 'optimal':
        nowait_states, nowait_actions = read_policy(n,p,p_s,cutoff,tolerance)

    #rng = np.random.RandomState(0)
    params=[n,p,p_s,cutoff,policy,max_distance]
    parameters = [copy.deepcopy(params) for s in range(N_samples)]
    parameters_with_seed = np.concatenate((parameters, seed_a), axis=1).tolist()
    #list_iterations=range(N_samples)
    #param=product([parameters],list_iterations)
    T_vec = [] # Array that stores delivery times of all samples
    
    #Create iteration batches depending on the number of available nodes
   
    
    with multiprocessing.Pool(processes=num_nodes) as pool:
        R_vec = pool.map(main.mc_iteration, parameters_with_seed)
        #print(T_vec[:][0])
    T_vec= [R[0] for R in R_vec]
    A_vec= [R[1] for R in R_vec]
    print(A_vec)
    # Store data as a histogram (for efficiency)
    bins_T = np.arange(-0.5, max(T_vec)+1.5)
    T_hist = np.histogram(T_vec, bins=bins_T)
    T_hist = T_hist[0]/sum(T_hist[0])
    T_values = np.arange(0, len(T_hist))
    bins_A = np.arange(-0.5, max(A_vec)+1.5)
    A_hist = np.histogram(A_vec, bins=bins_A)
    A_hist = A_hist[0]/sum(A_hist[0])
    A_values = np.arange(0, len(A_hist))
    data = {'avg_T': np.sum(np.multiply(T_values, T_hist)),
              'avg_A': np.sum(np.multiply(A_values, A_hist)),
              'std_T': np.std(T_vec),
              'std_A': np.std(A_vec),
              'N_samples': N_samples,
              'bins_T': bins_T,
              'bins_A': bins_A,
              'hist_T': T_hist,
              'hist_A': A_hist}
    
    if savedata:
        # Create data directory if needed
        try:
            os.mkdir('data_sim')
        except FileExistsError:
            pass

        # Save data
        filename = ('data_sim/%s_n%d_p%.3f_ps%.3f_tc%s'%(policy,n,p,p_s,cutoff)
                    +'_samples%d_randseed%s'%(N_samples,randomseed))
        if policy=='optimal':
            filename += '_tol%s.pickle'%tolerance
        else:
            filename += '.pickle'

        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print( data )  
    
