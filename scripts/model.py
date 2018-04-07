import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd._functions import stochastic as STO
from listmodule import ListModule

class Params():
    def __init__(self):
        self.batch_size = 64
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 2048
        self.time_horizon = 1000000
        self.max_episode_length = 10000
        self.seed = 1

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

def normal(x, mu, sigma_sq):
    a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*np.pi).sqrt()
    return a*b


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.saved_actions=[]
        self.saved_rewards_est=[]
        self.rewards=[]
        self.normal=STO.Normal()
        self.activ=nn.LeakyReLU()
        self.h0=Variable(torch.zeros(1,n_hidden), requires_grad=False)
        self.c0=Variable(torch.zeros(1,n_hidden), requires_grad=False)
        self.hi=[]
        self.ci=[]
        fc=[]
        lin=[]
        self.n_lstm=1
        self.n_lin=1
        for i in range(self.n_lstm):
            self.hi.append(Variable(torch.randn(1,n_hidden)))
            self.ci.append(Variable(torch.randn(1,n_hidden)))
            if(i==0):
                fc.append(nn.LSTMCell(n_input, n_hidden))
            else:
                fc.append(nn.LSTMCell(n_hidden, n_hidden))
        for i in range(self.n_lin):
            if(i<self.n_lin-1):
                lin.append(nn.Linear(n_hidden, n_hidden ))
            else:
                lin.append(nn.Linear(n_hidden, n_output ))
        self.fc=ListModule(*fc) 
        self.lin=ListModule(*lin)

    def reinit(self):
        for i in range(self.n_lstm):
            self.hi[i]=self.h0
            self.ci[i]=self.c0
    def forward(self, x):
        critic=x
        for i in range(self.n_lstm):
            self.hi[i], self.ci[i] = self.fc[i](x, (self.hi[i], self.ci[i]))
            x=self.ci[i]
        for i in range(self.n_lin):
            x=self.activ(self.lin[i](x))
            return x

class ValueNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ValueNet, self).__init__()
        critic_n=[]
        self.n_critic=10
        self.activ=nn.LeakyReLU()
        for i in range(self.n_critic):
            if(i==0):
                critic_n.append(nn.Linear(n_input, n_hidden ))
            elif(i<self.n_critic-1):
                critic_n.append(nn.Linear(n_hidden, n_hidden ))
            else:
                critic_n.append(nn.Linear(n_hidden, 1))

        self.critic_n=ListModule(*critic_n)

    def forward(self, x):
         critic=x
         for i in range(self.n_critic):
             critic=self.activ(self.critic_n[i](critic))
         return critic


class Agent: 
    def __init__(self,n_input,n_output):
        n_hidden=128
        self.pol=Net(n_input,256, n_output)
        self.val=ValueNet(n_input,256, 1)
        params=list(self.pol.parameters())+list(self.val.parameters())
        self.optimizer = optim.Adam(params, lr=1e-2)
        self.torque_int_std=0.1
        self.n_servo=6

    def forward(self,obs_input):
        pi=self.pol(obs_input)
        r_est=self.val(obs_input)
        action=torch.normal(Variable(pi.data[0,0:self.n_servo], requires_grad=True), Variable(self.torque_int_std*torch.abs(pi.data[0,self.n_servo:]), requires_grad=True))
        return action, r_est
