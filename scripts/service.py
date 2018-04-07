#!/usr/bin/env python3
import rospy
from threading import Thread
from ppo.srv import *
from model import *
from memory import *
from zfilter import *
from std_srvs.srv import Trigger
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

	def __init__(self):
		self.stop_client = rospy.ServiceProxy('/restartTrigger', Trigger)
		self.agent=Agent(12,12)
		num_inputs = 12
		self.running_state = ZFilter((num_inputs))
		self.running_reward = ZFilter((1))
		self.memory = Memory()
		self.n_steps = 100
		self.step_count = 0

	def handle_policy_query(self,req):
		if self.step_count != self.n_steps - 1:
			pi, val= self.agent.forward(Variable(torch.Tensor(self.running_state(list(req.obs)[:12])).unsqueeze(0)))
			self.step_count += 1
			return PolicyEvalResponse(tuple(pi)) 

		elif self.step_count == self.n_steps - 1:
			pi, val= self.agent.forward(Variable(torch.Tensor(self.running_state(list(req.obs)[:12])).unsqueeze(0)))
			self.step_count += 1
			thread = Thread(target=self.restart)
			thread.start()
			return PolicyEvalResponse(tuple(pi)) 

	def restart(self):
		self.stop_client()
		self.step_count = 0
		print("Simulation Stopped")

		

if __name__ == "__main__":
	service = PolicyService()
	rospy.init_node('policyServer')
	s = rospy.Service('policy_query', PolicyEval, service.handle_policy_query)
	print("Server Ready")
	rospy.spin()
