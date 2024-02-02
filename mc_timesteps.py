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
n=10
p=0.5
p_s=0.5
cutoff=10
policy="nested"
N_samples=10000
timesteps=1000
randomseed=1
seed_a=main.seed_array(randomseed,N_samples)
progress_bar='plain'
savedata=True
max_distance=2
num_nodes=7

## MC iteration definition

if __name__ == '__main__':
    np.random.seed(randomseed)
    random.seed(randomseed)
    if progress_bar == 'notebook':
        tqdm_ = tqdm_notebook
    elif progress_bar == 'plain':
        tqdm_ = tqdm
    else:
        raise ValueError('Invalid value for progress_bar.')

    if policy == 'optimal':
        nowait_states, nowait_actions = read_policy(n,p,p_s,cutoff,tolerance)

    #rng = np.random.RandomState(0)
    params=[n,p,p_s,cutoff,policy,max_distance,timesteps]
    parameters = [copy.deepcopy(params) for s in range(N_samples)]
    parameters_with_seed = np.concatenate((parameters, seed_a), axis=1).tolist()
    #list_iterations=range(N_samples)
    #param=product([parameters],list_iterations)
    T_vec = [] # Array that stores delivery times of all samples
    

        
    #Create iteration batches depending on the number of available nodes
    #iteration_batches = [list(range(start, start + N_samples // num_nodes)) for start in range(0, N_samples, N_samples // num_nodes)]
    
    with multiprocessing.Pool(processes=num_nodes) as pool:
        R_vec = pool.map(main.mc_timesteps, parameters_with_seed)
    
    prev_e2e=R_vec[0][0]
    prev_age=R_vec[0][1]
    prev_std_e2e=np.zeros(timesteps+1)
    prev_std_age=np.zeros(timesteps+1)
    print(prev_e2e)
    print(R_vec[1][0])
    print(R_vec[2][0])
    for i in range(N_samples):
        e2e=(R_vec[i][0]+i*prev_e2e)/(i+1)
        age=(R_vec[i][1]+i*prev_age)/(i+1)
        std_e2e=np.sqrt(i/(i+1)*prev_std_e2e**2+(i*(prev_e2e-R_vec[i][0])**2)/((i+1)**2))
        std_age=np.sqrt(i/(i+1)*prev_std_age**2+(i*(prev_age-R_vec[i][1])**2)/((i+1)**2))
        #print(e2e)
        prev_e2e=e2e
        prev_age=age
    #print(e2e)
    
    # Store data as a histogram (for efficiency)
    
   
    data = {'avg_R': e2e,
              'std_R': std_e2e ,
              'std_A': std_age,
              'avg_A': age,
              'N_samples': N_samples,
           }
    
    if savedata:
        # Create data directory if needed
        try:
            os.mkdir('data_sim')
        except FileExistsError:
            pass

        # Save data
        filename = ('data_sim/%s_n%d_p%.3f_ps%.3f_tc%d_timesteps%d'%(policy,n,p,p_s,cutoff,timesteps)
                    +'_samples%d_randseed%s'%(N_samples,randomseed))
        if policy=='optimal':
            filename += '_tol%s.pickle'%tolerance
        else:
            filename += '.pickle'

        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print( data )  
    
