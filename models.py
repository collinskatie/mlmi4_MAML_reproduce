'''
Houses main models used for experiments
'''

import torch.nn as nn
import torch

#Neural Network Class that Ocariz wrote
class Neural_Network(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=1, 
                    activation_func = "relu"):
        super(Neural_Network, self).__init__()
        # network layers
        self.hidden1 = nn.Linear(input_size,hidden_size)
        self.hidden2 = nn.Linear(hidden_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size,output_size)

        #Activation functions

        if activation_func == "ReLU": 
            self.nonlin = nn.ReLU()
        elif activation_func == "ELU": 
            self.nonlin = nn.ELU()
        elif activation_func == "SELU": 
            self.nonlin = nn.SELU()
        elif activation_func == "LeakyReLU": 
            self.nonlin = nn.LeakyReLU()

        self.model_tag = "baseline"
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.nonlin(x)
        x = self.hidden2(x)
        x = self.nonlin(x)
        x = self.output_layer(x)
        y = x
        return y


'''
Also learn to modulate the amplitude scale of the function
'''
class Neural_Network_Magnitude_Scaling(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=1):
        super(Neural_Network_Magnitude_Scaling, self).__init__()
        # network layers
        self.hidden1 = nn.Linear(input_size,hidden_size)
        self.hidden2 = nn.Linear(hidden_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size,output_size)
        self.modulation_layer = nn.Linear(output_size, output_size, bias=False)

        #Activation functions
        self.relu = nn.ReLU()

        self.model_tag = "mod"
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.modulation_layer(x)
        y = x
        return y


'''
Predict mean and variance 
'''
class Prob_Neural_Network(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=2):
        super(Prob_Neural_Network, self).__init__()
        # network layers
        self.hidden1 = nn.Linear(input_size,hidden_size)
        self.hidden2 = nn.Linear(hidden_size,hidden_size)
        self.hidden3 = nn.Linear(hidden_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size,output_size)

        #Activation functions
        self.relu = nn.ReLU()

        self.model_tag = "prob"
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output_layer(x)
        mean = x[:,0][:,None] 
        std = self.relu(x[:,1][:,None]) 
        return mean,std

'''
Predict gaussian mixture
'''
class Prob_Mixture_Neural_Network(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, number_of_gaussians=4):
        super(Prob_Mixture_Neural_Network, self).__init__()
        # network layers
        self.output_size = number_of_gaussians*3
        self.hidden1 = nn.Linear(input_size,hidden_size)
        self.hidden2 = nn.Linear(hidden_size,hidden_size)
        self.hidden3 = nn.Linear(hidden_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size,self.output_size)
        

        #Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output_layer(x)
        means = x[:,0:int(self.output_size/3)] 
        stds = self.relu(x[:,int(self.output_size/3):int(2*self.output_size/3)]) 
        weights = self.softmax(x[:,int(2*self.output_size/3):self.output_size]) 
        y = torch.cat((means,stds,weights),dim=-1)
        return y