import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class Memory():
	def __init__(self):
		self.memory = []
	def push(self, state, action, mask, next_state, reward):
		self.memory.append(Transition(state, action, mask, next_state, reward))
	def sample(self, n_samples):
		if mini_size > len(self.memory) :
			return Transition(*zip(*self.memory))
		else:
			return Transition(*zip(*np.random.choice(self.memory, n_samples)))
