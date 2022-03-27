'''
Main constants for training and testing parameters
Note, may be adjusted for specific use case
But includes defaults
Largely drawn from original MAML and Reptile papers
'''

from math import pi as PI

# using parameters from original MAML Section 5.1 (https://arxiv.org/pdf/1703.03400.pdf)
amp_min=0.1
amp_max=5.0
phase_min=0
phase_max=PI

# note: maml used 70k, but we find sufficient with 10k
num_epochs = 30000#10000
printing_step = 500 # print loss every x epochs

lr_task_specific = 0.01 # task specific learning rate
lr_meta = 0.001 # meta-update learning rate

T = 25 # num tasks per batch for MAML

val_batch_size = 25 # number of val waves per epoch
