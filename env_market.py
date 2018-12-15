import pyprocess as pp
import numpy as np

class state:
	A = 100.0
	B = 100.0

class mkt_env:
	s = state()
	index = 0
	timestep = 5001

	msg = "None"  #Info/Error

	def __init__(self):

		np.random.seed(10)
		self.s.A = 100.0
		self.s.B = 100.0
		self.theta = 3.0
		self.mean = 100.0
		self.vol_A = 10.0
		self.vol_B = 20.0

		#Now create the processes
		self.p1 = pp.OU_process(self.theta, self.mean, self.vol_A, 0, self.mean, None, None)
		self.p2 = pp.OU_process(self.theta, self.mean, self.vol_B, 0, self.mean, None, None)

		self.pa = np.zeros((5000, 1))
		self.pb = np.zeros((5000, 1))

	def reset(self):
		#Generate step from processes
		self.index = 0
		
		self.pa = self.p1.sample_path(range(self.timestep), 1)
		self.pa = self.pa.reshape(self.timestep,)

		self.pb = self.p2.sample_path(range(self.timestep), 1)
		self.pb = self.pb.reshape(self.timestep,)


	def step(self):
		#Returns 3 values

		done = False 

		self.s.A = self.pa[self.index]
		self.s.B = self.pb[self.index]
		self.msg = "None"

		self.index = self.index + 1

		if self.index > self.timestep - 1:
			done = True
			self.msg = "Episode over"

		return self.s, done, self.msg

