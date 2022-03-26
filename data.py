'''
Data loaders for sinusoid regression
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
            tasks = [SineFunction(amplitude = random.uniform(self.amp_min, self.amp_max), 
                                     phase=random.uniform(self.phase_min, self.phase_max)) for _ in range(num_tasks)]
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
    
    def __init__(self, amplitude, phase): 
        self.amplitude = amplitude
        self.phase = phase
        
    def draw_sample(self, x, with_noise=False, noise_dev=1): 
        '''
        Sample from the specified sine wave 
        '''
        # help to sample from a sine function:
        # https://stackoverflow.com/questions/48043004/how-do-i-generate-a-sine-wave-using-python
        freq = 1 # TODO: check???
        sample = self.amplitude * np.sin(freq * x + self.phase)
        
        if with_noise: 
            # corrupt sample w/ Gaussian noise
            # help from: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
            noise_corruption = np.random.normal(0, noise_dev)
            print("noise: ", noise_corruption, " noise_dev: ", noise_dev)
            sample += np.random.normal(0, noise_dev) # zero-mean
        
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
        

class SineWaveTask_multi:
    '''
    Multi-dimensional sine wave generator
    Note, structure and code are from  https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
    Custom modifications have been made for scaling to more dimensions, but base is from cited link
    '''
    def __init__(self,dimensions=20):
        self.dimensions = dimensions
        self.a = []
        self.b = []
        for dim in range(self.dimensions):
          self.a.append(np.random.uniform(0.1, 5.0))
          self.b.append(np.random.uniform(0, 2*np.pi))
        self.train_x = None
        
    def f(self, x,a,b):
        return a * np.sin(x + b)
        
    def training_set(self, size=10, force_new=False):
        if self.train_x is None and not force_new:
            self.train_x = np.random.uniform(-5, 5, size)
            x = self.train_x

        elif not force_new:
            x = self.train_x
        else:
            x = np.random.uniform(-5, 5, size)

        y = self.f(x,self.a[0],self.b[0])[:,None]

        for dim in range(self.dimensions-1):
          y = np.concatenate((y,self.f(x,self.a[dim+1],self.b[dim+1])[:,None]),axis=-1)

        return torch.Tensor(x[:,None]), torch.Tensor(y)
    
    def test_set(self, size=50):
        x = np.linspace(-5, 5, size)
        y = self.f(x,self.a[0],self.b[0])[:,None]

        for dim in range(self.dimensions-1):
          y = np.concatenate((y,self.f(x,self.a[dim+1],self.b[dim+1])[:,None]),axis=-1)

        return torch.Tensor(x[:,None]), torch.Tensor(y)


if __name__ == "main": 

    '''
    Sample of how to use the custom-written unidimensional dataset
    '''

    # using parameters from original MAML Section 5.1 (https://arxiv.org/pdf/1703.03400.pdf)
    amp_min=0.1
    amp_max=5.0
    phase_min=0
    phase_max=PI
    K = 10

    # todo: check parameters we want
    # specify the number of tasks to sample per meta-set
    meta_train_size=1000
    meta_val_size=100
    meta_test_size=1000
    meta_train_eval_size = 20

    task_batch_size = 10  

    dataset = RegressionDomain(amp_min=amp_min, amp_max=amp_max, 
                            phase_min=phase_min, phase_max=phase_max, 
                            train_size=meta_train_size, val_size=meta_val_size, test_size=meta_test_size)

    meta_val_set = dataset.get_meta_val_batch()
    meta_test_set = dataset.get_meta_test_batch()

    meta_train_sample = dataset.get_meta_train_batch(task_batch_size=task_batch_size)

    meta_train_sample[0].get_samples()