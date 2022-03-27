'''
Houses main util functions for training and evaluation algs
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

from data import *
from models import Neural_Network

# set GPU or CPU depending on available hardware
# help from: https://stackoverflow.com/questions/46704352/porting-pytorch-code-from-cpu-to-gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device}")

if device == "cuda:0": 
  # set default so all tensors are on GPU, if available
  # help from: https://stackoverflow.com/questions/46704352/porting-pytorch-code-from-cpu-to-gpu
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_samples_in_good_format(wave, num_samples=10, with_noise=False, noise_dev=1, input_range=[-5.0, 5.0]):
    #This function is used to sample data from a wave
    sample_data = wave.get_samples(num_samples=num_samples,with_noise=with_noise, noise_dev=noise_dev,
                        min_query_x=input_range[0], max_query_x=input_range[1])
    x = sample_data["input"]
    y_true = sample_data["output"]
    # We add [:,None] to get the right dimensions to pass to the model: we want K x 1 (we have scalars inputs hence the x 1)
    # Note that we convert everything torch tensors
    x = torch.tensor(x[:,None])
    y_true = torch.tensor(y_true[:,None])
    # set to whatever the base device is (GPU or CPU)
    # help from: https://stackoverflow.com/questions/46704352/porting-pytorch-code-from-cpu-to-gpu
    return x.to(device),y_true.to(device) 


def copy_existing_model(model):
    # Function to copy an existing model
    # We initialize a new model
    new_model = Neural_Network()
    # Copy the previous model's parameters into the new model
    new_model.load_state_dict(model.state_dict())
    return new_model

def training_reptile(model, wave, criterion, lr_k, k):
    # Create new model which we will train on
    new_model = copy_existing_model(model)
    # Define new optimizer
    koptimizer = torch.optim.SGD(new_model.parameters(), lr=lr_k)
    # Update the model multiple times, note that k>1 (do not confuse k with K)
    for i in range(k):
        # Reset optimizer
        koptimizer.zero_grad()
        # Evaluate the model
        loss = evaluation(new_model, wave,criterion)
        # Backpropagate
        loss.backward()
        koptimizer.step()
    return new_model, loss

def metaupdate(model,new_model,metaoptimizer):
  # Combine the two previous functions into a single metaupdate function
  # First we calculate the gradients
  reptile_parameter_update(model,new_model)
  # Use those gradients in the optimizer
  metaoptimizer_update(metaoptimizer)

def test_set_validation(model,new_model,wave,criterion,lr_inner,k):
    # This functions does not actually affect the main algorithm, it is just used to evaluate the new model
    new_model, oldLoss = training_reptile(model, wave, criterion, lr_inner, k)
    # Obtain the loss
    loss = evaluation(new_model, wave, criterion)
    # Store loss
    return loss


def evaluation(new_model, wave, criterion, num_samples=1):
    # Get data
    x, label = get_samples_in_good_format(wave,num_samples=num_samples)
    # Make model prediction
    prediction = new_model(x)
    # Get loss
    return criterion(prediction,label)

def reptile_parameter_update(model,new_model):
  # Zip models for the loop
  zip_models = zip(model.parameters(), new_model.parameters())
  for parameter, new_parameter in zip_models:
    if parameter.grad is None:
      parameter.grad = torch.tensor(torch.zeros_like(parameter))
    # Here we are adding the gradient that will later be used by the optimizer
    parameter.grad.data.add_(parameter.data - new_parameter.data)

# Define commands in order needed for the metaupdate
# Note that if we change the order it doesn't behave the same
def metaoptimizer_update(metaoptimizer):
  # Take step
  metaoptimizer.step()
  # Reset gradients
  metaoptimizer.zero_grad()

'''
Handling computation graphs and second-order backprop help and partial inspiration from: 
- https://discuss.pytorch.org/t/how-to-save-computation-graph-of-a-gradient/128286/2 
- https://discuss.pytorch.org/t/when-do-i-use-create-graph-in-autograd-grad/32853/3 
- https://lucainiaoge.github.io/download/PyTorch-create_graph-is-true_Tutorial_and_Example.pdf
- https://www.youtube.com/watch?v=IkDw22a8BDE
- https://discuss.pytorch.org/t/how-to-manually-update-network-parameters-while-keeping-track-of-its-computational-graph/131642/2
- https://discuss.pytorch.org/t/how-to-calculate-2nd-derivative-of-a-likelihood-function/15085/3
- https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
- https://higher.readthedocs.io/en/latest/toplevel.html

Neural network configuration and helper class functions copied directly from 
-https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb

Note, different ways to refer to the task-specific vs. meta/aggregate updates to the parameters
Sometimes called "inner" and "outer" loop, respectively
Here, refered to as "task_specific" and "agg"/meta" (the latter, for consistency w/ ocariz code)
'''
def task_specific_train_and_eval(model, T_i, inner_loop_optimizer, criterion, K = 10, N=1, extract_task_info=False,
                                with_noise=False, noise_dev=1 ,input_range=[-5.0,5.0]):
    
    '''
    if extract_task_info is True => return information on the initial task, and intermediate preds
    don't save for storage and compute reasons if set to False (by default)
    '''
    
    x, label = get_samples_in_good_format(T_i,num_samples=K,with_noise=with_noise, noise_dev=noise_dev, input_range=input_range)

    task_info = {}
    
    if extract_task_info == True: 
        input_coords = x.detach().numpy()[:,0]
        true_vals = sorted([(x, y) for (x, y) in zip(input_coords, label)], key=lambda x: x[0])
        
        task_info["input_coords"] = input_coords
        task_info["gt"] = true_vals
        
        # generate more points for a fine-grained evaluation of underlying func
        eval_x, eval_true_y = get_samples_in_good_format(T_i,num_samples=10000, input_range=input_range)
        
        eval_coords = eval_x.detach().numpy()[:,0]
        task_info["eval_coords"] = eval_coords
        task_info["gt_func"] = sorted([(x, y) for (x, y) in zip(eval_coords, eval_true_y)], key=lambda x: x[0])
        
    #Description of the loop formulation from https://higher.readthedocs.io/en/latest/toplevel.html
    with innerloop_ctx(model, inner_loop_optimizer, copy_initial_weights = False) as (fmodel,diffopt):
        #get our input data and our label
        per_step_loss = []
        for grad_step in range(N):
            
            if extract_task_info: 
                # use preds for new points
                preds_eval = fmodel(eval_x)
                task_preds = preds_eval.detach().numpy()[:,0]
                pred_data = sorted([(x, y) for (x, y) in zip(eval_coords, task_preds)], key=lambda x: x[0])
                task_info[f"preds_{grad_step}"] = pred_data
            
            preds = fmodel(x)
            #Get the task specific loss for our model
            task_specifc_loss = criterion(preds, label)

            #Step through the inner gradient
            diffopt.step(task_specifc_loss)
            
            per_step_loss.append(task_specifc_loss.item())
            
        held_out_task_specific_loss = evaluation(fmodel, T_i, criterion, num_samples=K)
        
        return held_out_task_specific_loss, per_step_loss, fmodel, task_info