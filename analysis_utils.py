'''
Main analysis and plotting util functions 
Seeds are set for reproducibility and being able to extract representative results for the report
We therefore note that results are somewhat hand-selected
If needed, a diverse range of behavior can be seen by changing seeds, but we always tried to have results which demonstrate typical behavior
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns 
from math import pi as PI
import random
# !pip3 install higher
import torch.nn as nn
import torch
import random
from higher import innerloop_ctx
import warnings

from constants import *
from utils import task_specific_train_and_eval, test_set_validation

import pickle


ax_size = 14
title_size = 16

dpi=800

results_dir = "./report_plots/"
backup_dir = "./backup_data/"

def k_shot_evaluation(model, dataset, criterion, num_k_shots=10, K=10, num_eval=100,
                        file_tag="maml", seed=11, analysis_steps=[0,1,5], input_range=[-5.0,5.0],
                        use_reptile=False): 

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    all_losses = []
    test_waves = dataset.get_meta_test_batch(task_batch_size=num_eval)

    for test_eval in range(num_eval): 
        test_wave = test_waves[test_eval]

        if use_reptile: 
            metaTrainLosses = test_set_validation(model,None,test_wave,criterion,0.01,num_k_shots)
            # expand dim help from: https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array
            # all_losses.append(np.array(metaTrainLosses.detach().numpy())[..., np.newaxis])
            all_losses.append(np.array(metaTrainLosses))
        else: 
            # use model returned from earlier optimization
            inner_loop_optimizer = torch.optim.SGD(model.parameters(), lr = lr_task_specific)
            held_out_task_specific_loss, metaTrainLosses, _, _ = task_specific_train_and_eval(model, test_wave, inner_loop_optimizer, criterion, K=K, N=num_k_shots ,input_range=input_range)
            all_losses.append(np.array(metaTrainLosses))

    all_losses = np.array(all_losses)
    print("shape: ", np.shape(all_losses))
    np.save(f"{backup_dir}k_shot_{file_tag}.npy", all_losses)
        
    fig, ax = plt.subplots(figsize=(8,4))

    mean_loss = np.mean(all_losses, axis=0)

    # confidence interval plotting help from: https://stackoverflow.com/questions/59747313/how-to-plot-confidence-interval-in-python
    y = mean_loss
    x = list(range(num_k_shots))
    ci = 1.96 * np.std(all_losses, axis=0)**2/np.sqrt(len(y))

    ax_size=16
    title_size=18
                                                    
    ax.plot(x, y, linewidth=3, label=f"Mean Loss")
    # to avoid having MSE < 0
    truncated_error = np.clip(y-ci, a_min=0, a_max=None)
    ax.fill_between(x, truncated_error, (y+ci), alpha=.5,label=f"95% CI")
    ax.set_xlabel("Gradient Steps",fontsize=ax_size)
    ax.set_ylabel("Mean Squared Error (MSE)",fontsize=ax_size)
    ax.set_title("Sine Wave Regression: k-Shot Evaluation",fontsize=title_size)
    ax.legend(fontsize=15)#loc="upper right")
    plt.savefig(f"{results_dir}k_shot_{file_tag}.pdf",dpi=600, bbox_inches="tight")

    # analysis_steps = [0, 1, 5, num_k_shots-1]
    for analysis_step in analysis_steps: 
        print(f"Step: {analysis_step}, Error: {mean_loss[analysis_step]}, Var: {ci[analysis_step]}")

    return all_losses

# def k_shot_eval_reptile(model, dataset, criterion, num_k_shots=10, K=10, num_eval=100,
#                         file_tag="reptile", seed=11, analysis_steps=[0,1,5], input_range=[-5.0,5.0],
#                         lr_k_reptile=0.01): 

#     all_losses = []

#     test_waves = dataset.get_meta_test_batch(task_batch_size=num_eval)
#     for test_eval in range(num_eval): 
#         test_wave = test_waves[test_eval]
#         metaTrainLosses = test_set_validation(model,None,test_wave,criterion,lr_k_reptile,num_k_shots)
#         all_losses.append(np.array(metaTrainLosses.detach().numpy()))

#     all_losses = np.array(all_losses)
    
#     np.save(f"{backup_dir}k_shot_{file_tag}.npy", all_losses)
        
#     fig, ax = plt.subplots(figsize=(8,4))

#     mean_loss = np.mean(all_losses, axis=0)

#     # confidence interval plotting help from: https://stackoverflow.com/questions/59747313/how-to-plot-confidence-interval-in-python
#     y = mean_loss
#     x = list(range(num_k_shots))
#     ci = 1.96 * np.std(all_losses, axis=0)**2/np.sqrt(len(y))
                                                    
#     ax.plot(x, y, linewidth=3, label=f"Mean Loss")
#     # to avoid having MSE < 0
#     truncated_error = np.clip(y-ci, a_min=0, a_max=None)
#     ax.fill_between(x, truncated_error, (y+ci), alpha=.5,label=f"95% CI")
#     ax.set_xlabel("Gradient Steps",fontsize=ax_size)
#     ax.set_ylabel("Mean Squared Error (MSE)",fontsize=ax_size)
#     ax.set_title("Sine Wave Regression: k-Shot Evaluation",fontsize=title_size)
#     ax.legend(fontsize=15)#loc="upper right")
#     plt.savefig(f"{results_dir}k_shot_{file_tag}.pdf",dpi=600, bbox_inches="tight")

#     # analysis_steps = [0, 1, 5, num_k_shots-1]
#     for analysis_step in analysis_steps: 
#         print(f"Step: {analysis_step}, Error: {mean_loss[analysis_step]}, Var: {ci[analysis_step]}")

#     return all_losses

def plot_training_dynamics(metaLosses,metaValLosses, file_tag="maml"): 
    '''
    Display train and validation loss during learning
    '''
    num = 20
    avgTrain = []
    avgVal = []
    for r in range(int(num/2),int(len(metaLosses)-num/2)):
        currSum1 = 0
        for t in range(int(-num/2),int(num/2)):
            currSum1 += metaLosses[r+t]
        currSum1 /= num
        avgTrain.append(currSum1)
        
        currSum2 = 0
        for s in range(int(-num/2),int(num/2)):
            currSum2 += metaValLosses[r+s]
        currSum2 /= num
        avgVal.append(currSum2)
        
        
    plt.plot(avgVal) 
    plt.plot(avgTrain) 
    plt.legend(['Validation Loss','Train Loss'])
    plt.savefig(f"{results_dir}training_dynamics_{file_tag}.pdf",dpi=dpi, bbox_inches="tight")


def compare_K_shot(model, dataset, criterion, K_vals = [5,10], num_k_shots=10, seed=11, file_tag="maml",
                    title="MAML K-Shot Learning Comparison", plot_lims=[-5.0, 5.0], input_range=[-5.0, 5.0],
                    legend_locs=["upper right", "upper right"]):
    '''
    Compare fitting to functions with varied K shots
    Following MAML Fig. 2 structure: https://arxiv.org/pdf/1703.03400.pdf
    ''' 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    fig.suptitle(title, fontsize=title_size + 3)

    for idx, K in enumerate(K_vals): 
        ax=axes[idx]
        
        test_wave = dataset.get_meta_test_batch(task_batch_size=1)[0]
        # use model returned from earlier optimization
        inner_loop_optimizer = torch.optim.SGD(model.parameters(), lr = lr_task_specific)
        held_out_task_specific_loss, metaTrainLosses, _, task_info = task_specific_train_and_eval(model, test_wave, inner_loop_optimizer, criterion, K, num_k_shots, extract_task_info=True, input_range=input_range)

        # saving help from: https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
        with open(f'{backup_dir}{file_tag}_Kshot_{K}_plot_data.pkl', 'wb') as f:
            pickle.dump(task_info, f)
        '''
        Can be loaded back via the following
        Help from: https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
        with open(f'maml_{K}_plot_data.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        '''
        
        true_vals = task_info["gt"]
        func_gt = task_info["gt_func"]
        func_prior = task_info["preds_0"]
        updated_func_single = task_info["preds_1"]
        updated_func_many = task_info[f"preds_{num_k_shots-1}"]

        ax.plot(np.array(func_gt)[:,0], np.array(func_gt)[:,1], label="Ground Truth")
        ax.plot(np.array(true_vals)[:,0], np.array(true_vals)[:,1], '*', label="Observations", color="blue")
        ax.plot(np.array(func_prior)[:,0], np.array(func_prior)[:,1], label="Pre-Update")
        ax.plot(np.array(updated_func_single)[:,0], np.array(updated_func_single)[:,1], label="1 Gradient Step")
        ax.plot(np.array(updated_func_many)[:,0], np.array(updated_func_many)[:,1], label=f"{num_k_shots-1} Gradient Steps")
        
        ax.set_xlabel("",fontsize=ax_size)
        ax.set_ylabel("",fontsize=ax_size)
        ax.set_title(f"K={K}",fontsize=title_size)
        ax.set_ylim(plot_lims)
        ax.legend(loc=legend_locs[idx],fontsize=10)

    plt.savefig(f"{results_dir}Kshot_{file_tag}.pdf",dpi=dpi, bbox_inches="tight")