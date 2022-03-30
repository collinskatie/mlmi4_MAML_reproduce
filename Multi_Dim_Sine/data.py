
'''
Data loaders for sinusoid regression with multiple dimensional sines
Heavily borrowed and modified from: https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
'''

import numpy as np
import torch
from math import pi as PI

# using parameters from original MAML Section 5.1 (https://arxiv.org/pdf/1703.03400.pdf)
amp_min=0.1
amp_max=5.0
phase_min=0
phase_max=PI

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
          self.a.append(np.random.uniform(amp_min, amp_max))
          self.b.append(np.random.uniform(phase_min, phase_max))
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