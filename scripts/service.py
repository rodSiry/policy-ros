#!/usr/bin/env python3
import rospy
from threading import Thread
from ppo.srv import *
from model import *
from memory import *
from zfilter import *
from std_srvs.srv import Trigger
from utils import *
import random
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd._functions import stochastic as STO
from time import sleep



class PolicyService():

	def __init__(self, load=""):

		self.start_client = rospy.ServiceProxy('/startTrigger', Trigger)
		self.stop_client = rospy.ServiceProxy('/stopTrigger', Trigger)
		self.policy_net = PolicyNet(15,128,6)
		self.value_net = ValueNet(15,128,1)
		self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=0.001)
		self.opt_value = optim.Adam(self.value_net.parameters(), lr=0.001)
		if not load == "":
			state = torch.load(load)
			self.policy_net.load_state_dict(state['policy_net'])
			self.value_net.load_state_dict(state['value_net'])
			self.opt_policy.load_state_dict(state['opt_policy'])
			self.opt_value.load_state_dict(state['opt_value'])

		self.args = Params()
		num_inputs = 15
		msg_size = 16
		self.n_steps = 1000
		self.batch_size = 5000

		self.running_state = ZFilter((msg_size))
		self.running_reward = ZFilter((1))
		self.memory = Memory()
		self.num_epoch = 0
		self.num_episodes = 0
		self.step_count_ep = 0
		self.step_count = 0
		self.start_client()

	def select_action(self, state):
		action_mean, action_log_std, action_std, action_mean_old, action_log_std_old, action_std_old = self.policy_net(Variable(torch.Tensor(self.state).unsqueeze(0)))
		action = torch.normal(action_mean, action_std)
		return action, action_mean, action_log_std, action_std, action_mean_old, action_log_std_old, action_std_old

	def handle_policy_query(self,req):

		req = self.running_state(list(req.obs))
		self.state  = req[:15]
		self.reward = req[15]

		if self.step_count_ep < self.n_steps - 1:
			pi = self.select_action(self.state)
			if self.step_count_ep > 0 :
				self.memory.push(self.prev_state, *self.prev_action, 1, self.state, self.reward)
			self.prev_state = self.state
			self.prev_action = pi
			self.step_count_ep += 1
			return PolicyEvalResponse(tuple(pi[0][0]))

		elif self.step_count_ep == self.n_steps - 1:
			pi = self.select_action(self.state)
			self.step_count += self.step_count_ep
			self.step_count_ep += 1
			thread = Thread(target=self.restart_episode)
			thread.start()
			return PolicyEvalResponse(tuple(pi[0][0])) 

		else:
			pi = self.select_action(self.state)
			return PolicyEvalResponse(tuple(pi[0][0])) 

	def update_params(self):
		batch = self.memory.sample()
		rewards = torch.Tensor(batch.reward)
		masks = torch.Tensor(batch.mask)
		states = torch.Tensor(batch.state)
		actions = torch.cat(list(batch.action), dim=0)
		values = self.value_net(Variable(states))
		returns = torch.Tensor(actions.size(0),1)
		deltas = torch.Tensor(actions.size(0),1)
		advantages = torch.Tensor(actions.size(0),1)

		prev_return = 0
		prev_value = 0
		prev_advantage = 0
		
		for i in reversed(range(rewards.size(0))):
			returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
			deltas[i] = rewards[i] + self.args.gamma * prev_value * masks[i] - values.data[i]
			advantages[i] = deltas[i] + self.args.gamma * self.args.tau * prev_advantage * masks[i]
			prev_return = returns[i, 0]
			prev_value = values.data[i, 0]
			prev_advantage = advantages[i, 0]

		targets = Variable(returns)
		self.opt_value.zero_grad()
		value_loss = (values - targets).pow(2.).mean()
		print(targets)
		value_loss.backward()
		self.opt_value.step()
		action_var = Variable(actions.clone())

		action_means = torch.cat((batch.action_mean), 0)
		action_std = torch.cat((batch.action_std), 0)
		action_log_std = torch.cat((batch.action_log_std), 0)

		action_means_old = torch.cat((batch.action_mean_old), 0)
		action_std_old = torch.cat((batch.action_std_old), 0)
		action_log_std_old = torch.cat((batch.action_log_std_old), 0)

		log_prob_cur = normal_log_density(action_var, action_means, action_log_std, action_std)
		log_prob_old = normal_log_density(action_var, action_means_old, action_log_std_old, action_std_old)
		advantages = (advantages - advantages.mean()) / advantages.std()
		advantages_var = Variable(advantages)
		self.policy_net.backup()
		self.opt_policy.zero_grad()
		ratio = torch.exp(log_prob_cur - log_prob_old)
		surr1 = ratio * advantages_var[:,0]
		surr2 = torch.clamp(ratio, 1.0 - self.args.clip, 1.0 + self.args.clip) * advantages_var[:,0]
		policy_surr =  -torch.min(surr1, surr2).mean()
		policy_surr.backward()
		torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 40)
		self.opt_policy.step()
		torch.save({'policy_net':self.policy_net.state_dict(), 'opt_policy':self.opt_policy.state_dict(), 'value_net':self.value_net.state_dict(), 'opt_value':self.opt_value.state_dict()}, '/home/rodrigue/catkin_ws/src/ppo/scripts/model/model.pt')
		print("fin training")

	def restart_episode(self):

		if self.step_count < self.batch_size:
			self.stop_client()
			self.step_count_ep = 0
			self.num_episodes +=1
			self.policy_net.reinit()
			self.start_client()
		else:
			self.stop_client()
			self.update_params()
			self.num_epoch +=1
			print(self.num_epoch)
			self.memory = Memory()
			self.step_count_ep = 0
			self.step_count = 0
			self.num_episodes = 0
			self.policy_net.reinit()
			self.start_client()

		

if __name__ == "__main__":

	service = PolicyService('/home/rodrigue/catkin_ws/src/ppo/scripts/model/model.pt')
	rospy.init_node('policyServer')
	s = rospy.Service('policy_query', PolicyEval, service.handle_policy_query)
	print("Server Ready")
	rospy.spin()
