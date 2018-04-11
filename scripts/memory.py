import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'action_mean', 'action_log_std', 'action_std', 'action_mean_old', 'action_log_std_old', 'action_std_old', 'mask', 'next_state', 'reward'))

class Memory():
	def __init__(self):
		self.memory = []
	def push(self, state, action, action_mean, action_log_std, action_std, action_mean_old, action_log_std_old, action_std_old, mask, next_state, reward):
		self.memory.append(Transition(state, action.data, action_mean, action_log_std, action_std, action_mean_old, action_log_std_old, action_std_old, mask, next_state, reward))
	def sample(self):
		return Transition(*zip(*self.memory))
