'''
Data loader for sinusoid regression
But with complex noise distribution 
Note, this is very similar to data.py
Keeping in a separate file to avoid inducing any issues in other code
'''

import numpy as np
import pandas as pd
from math import pi as PI
import random
# !pip3 install higher
import torch
import random

#Set random seeds for reproducibility of results 
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

'''
Create a master data class to ensure analogous structure across tasks
Help on writing abstract classes from: 
https://www.geeksforgeeks.org/abstract-classes-in-python/

Meta-train has training and test sets
Also have meta-val and meta-test sets (at the task level)
Help for understanding training and test set structure from [see especially for classification]: 
https://meta-learning.fastforwardlabs.com/


What do we want out of our data generation code? 
- specify a particular task instance
(e.g., amplitude and phase; N number of discrete classes)
- extract batches of tasks for train, val, and test
- extract sample from a given task of size K for training (and other for testing)
'''

class Domain(): 
    
    def get_meta_train_batch(self, task_batch_size):
        # yields the set of meta-training tasks, each of which has train and test sets
        pass 
    
    def get_meta_val_batch(self, task_batch_size):
        # yields meta-val tasks (each just has a single data set)
        pass

    def get_meta_test_batch(self, task_batch_size):
        # yields meta-test tasks (each just has a single data set)
        pass
'''
Regression task, as per MAML Section 5.1 (https://arxiv.org/pdf/1703.03400.pdf)
Specifically, sine wave generation 

Code inspired by and modified from: 
https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb

Checked original MAML code to ensure resampling for all stages of sinusoid:
https://github.com/cbfinn/maml/blob/master/data_generator.py
'''

class RegressionDomain(Domain): 
    
    '''
    Each task is a sine wave
    Parameterized by amplitude and phase 
    Always drawn from w/in a specified range of x vals
    [Values from Section 5.1 -- but we could vary??]
    '''
    
    def __init__(self, amp_min=0.1, amp_max=0.5, 
                 phase_min=0, phase_max=PI,
                train_size=1000, val_size=100, test_size=1000): 
        
        self.amp_min = amp_min
        self.amp_max = amp_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        
        # create initial train, val, and test 
        # parameters specify the number of unique functions we want
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        # looping to instantiate tasks idea from: https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
        # help on generating random numbers in range from: https://pynative.com/python-get-random-float-numbers/
        self.tasks = {}
        # note: code would be more structured b/w task type for classification
        for task_type, num_tasks in zip(["train", "val", "test"], [train_size, val_size, test_size]):
            tasks = [SineFunction(amplitude1 = random.uniform(self.amp_min, self.amp_max), 
                                     phase1=random.uniform(self.phase_min, self.phase_max),
                                     amplitude2 = random.uniform(self.amp_min, self.amp_max), 
                                     phase2 = random.uniform(self.phase_min, self.phase_max), 
                                     weighting = random.uniform(0.000001, 0.99999)) for _ in range(num_tasks)]
            self.tasks[task_type] = tasks
    
    def get_batch_of_tasks(self, task_type, task_batch_size): 
        # helper function since same sampling per type for regression-specific domain
        if task_batch_size is None: 
            # return all 
            return self.tasks[task_type]
        else: 
            # sub-sample
            # note: we could investigate impact of weighted sub-sampling in batch (?)
            # see documentation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            task_batch = np.random.choice(self.tasks[task_type], size=task_batch_size, replace=False)
            return task_batch
    
    def get_meta_train_batch(self, task_batch_size=10): 
        return self.get_batch_of_tasks("train", task_batch_size)
        
    def get_meta_val_batch(self, task_batch_size=None): 
        return self.get_batch_of_tasks("val", task_batch_size) 
        
    def get_meta_test_batch(self, task_batch_size=None): 
        return self.get_batch_of_tasks("test", task_batch_size)

class SineFunction(): 
    
    def __init__(self, amplitude1, phase1, amplitude2, phase2, weighting): 
        self.amplitude1 = amplitude1
        self.phase1 = phase1

        self.amplitude2 = amplitude2 
        self.phase2 = phase2

        self.weighting = weighting
        
    def draw_sample(self, x, with_noise=False, noise_dev=1): 
        '''
        Sample from the specified sine wave 
        '''
        # help to sample from a sine function:
        # https://stackoverflow.com/questions/48043004/how-do-i-generate-a-sine-wave-using-python
        freq = 1 
        sample1 = self.amplitude1 * np.sin(freq * x + self.phase1)

        sample2 = self.amplitude2 * np.sin(freq * x + self.phase2)
        
        if with_noise: 

            # simulate more complex noise 
            # gaussian corruption help from 
            # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python

            # assume gaussian "world" noise over entire domain 
            noise1 =  np.random.normal(0, 0.1)

            noise2 = np.random.normal(0, 0.3)

            sample1 += noise1 
            sample2 += noise2 

        sample = (sample1 * self.weighting) + sample2 * (1-self.weighting)
        
        return sample
    
    def get_samples(self, num_samples=10, 
                    min_query_x=-5.0, max_query_x=5.0,
                   with_noise=False,noise_dev=1): 
        '''
        Return samples drawn from this specific function (e.g., K for training set in meta-train)
        Note, input range uses values from paper (Section 5.1)
        But modification allowed thru function so we can test generalization beyond??
        '''
        x_vals = [random.uniform(min_query_x, max_query_x) for _ in range(num_samples)]
        y_vals = [self.draw_sample(x,with_noise=with_noise,noise_dev=noise_dev) for x in x_vals]
        # conversion to tensor idea and code help from: https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
        return {"input": torch.Tensor(x_vals), "output": torch.Tensor(y_vals)}