import torch 
import copy
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
        self.tau = 0.97
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 2048
        self.time_horizon = 1000000
        self.max_episode_length = 10000
        self.seed = 1

class PolicyNet(nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		super(PolicyNet, self).__init__()
		self.normal = STO.Normal()
		self.activ = nn.LeakyReLU()
		self.h0 = Variable(torch.zeros(1,n_hidden), requires_grad=False)
		self.c0 = Variable(torch.zeros(1,n_hidden), requires_grad=False)
		self.action_log_std = nn.Parameter(torch.zeros(1, n_output))
		self.hi = []
		self.ci = []
		self.hi_old = []
		self.ci_old = []
		fc = []
		lin = []
		self.n_lstm = 1
		self.n_lin = 1
		for i in range(self.n_lstm):
			self.hi.append(self.h0)
			self.hi_old.append(self.h0)
			self.ci.append(self.c0)
			self.ci_old.append(self.c0)
			if(i==0):
				fc.append(nn.LSTMCell(n_input, n_hidden))
			else:
				fc.append(nn.LSTMCell(n_hidden, n_hidden))
		for i in range(self.n_lin):
			if(i<self.n_lin-1):
				lin.append(nn.Linear(n_hidden, n_hidden ))
			else:
				lin.append(nn.Linear(n_hidden, n_output ))
		self.fc = ListModule(*fc) 
		self.lin = ListModule(*lin)
		self.module_list_current = [self.fc, self.lin, self.action_log_std]
		self.module_list_old = [None] * len(self.module_list_current)
		self.backup()
		self.fc_old, self.lin_old, self.action_log_std_old = tuple(self.module_list_old)

	def reinit(self):
		for i in range(self.n_lstm):
			self.hi[i] = self.h0
			self.hi_old[i] = self.h0
			self.ci[i] = self.c0
			self.ci_old[i] = self.c0

	def backup(self):
		for i in range(len(self.module_list_current)):
			self.module_list_old[i] = copy.deepcopy(self.module_list_current[i])

	def forward(self, x):
		x_old = x
		for i in range(self.n_lstm):
			self.hi[i], self.ci[i] = self.fc[i](x, (self.hi[i], self.ci[i]))
			x = self.ci[i]
		for i in range(self.n_lin):
			action_mean =self.activ(self.lin[i](x))
		action_log_std = self.action_log_std.expand_as(action_mean)
		action_std = torch.exp(action_log_std)

		self.fc_old, self.lin_old, self.action_log_std_old = tuple(self.module_list_old)
		for i in range(self.n_lstm):
			self.hi_old[i], self.ci_old[i] = self.fc_old[i](x_old, (self.hi_old[i], self.ci_old[i]))
			x_old = self.ci_old[i]
		for i in range(self.n_lin):
			action_mean_old =self.activ(self.lin_old[i](x_old))
		action_log_std_old = self.action_log_std_old.expand_as(action_mean_old)
		action_std_old = torch.exp(action_log_std_old)

		return action_mean, action_log_std, action_std, action_mean_old.detach(), action_log_std_old.detach(), action_std_old.detach()

class ValueNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ValueNet, self).__init__()
        critic_n=[]
        self.n_critic=5
        for i in range(self.n_critic):
            if(i==0):
                critic_n.append(nn.Linear(n_input, n_hidden ))
            elif(i<self.n_critic-1):
                critic_n.append(nn.Linear(n_hidden, n_hidden ))
            else:
                critic_n.append(nn.Linear(n_hidden, 1))

        self.critic_n=ListModule(*critic_n)

    def forward(self, x):
         state_value = x
         for i in range(self.n_critic-1):
             state_value = F.tanh(self.critic_n[i](state_value))
         state_value = self.critic_n[self.n_critic-1](state_value)
         return state_value
