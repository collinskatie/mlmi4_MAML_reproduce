'''
Main constants for training and testing parameters
Note, may be adjusted for specific use case
But includes defaults
Largely drawn from original MAML and Reptile papers
'''

from math import pi as PI

amp_min=0.1
amp_max=5.0
phase_min=0
phase_max=PI

# note: maml used 70k, but we find sufficient with 10k
num_epochs = 10000
printing_step = 500 # print loss every x epochs

lr_task_specific = 0.01 # task specific learning rate
lr_meta = 0.01 # meta-update learning rate
