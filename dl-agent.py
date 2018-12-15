import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import sgd
from collections import deque
import random

from env_market import *

steps = 500

class Agent:

	def __init__(self):
		self.memory = deque(maxlen=500000)
		self.learning_rate = 0.001
		self.gamma = 0.9
		self.exploration_rate = 1.0
		self.exploration_min = 0.01
		self.exploration_decay = 0.95
		self.brain	= self.build_model()

		self.total_reward = 0.0 #Reward at the end of every episode

		self.apos = 0
		self.bpos = 0

	def build_model(self):

		#input: price of A and B
		#output: 0|1 for A and 0|1 for B where 0 => buy and 1 => sell

		model = Sequential()
		model.add(Dense(20, input_dim = 1, activation = 'relu'))
		model.add(Dense(20, activation='relu'))
		model.add(Dense(20, activation='relu'))
		model.add(Dense(3, activation='softmax'))
		model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
		
		print "Model Created"

		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))	

	
	def replay(self, batch_size):
		if (len(self.memory) < batch_size):
			return

		sample_batch = random.sample(self.memory, batch_size)

		#Online training with this sample
		for state, action, reward, next_state, done in sample_batch:
			if done: #End of episode
				target = reward
			else:
				target_f = self.brain.predict(np.array([[state.A - state.B]]))
				target = reward + self.gamma * np.amax(self.brain.predict(np.array([next_state.A - next_state.B]))[0])
		
				y = np.zeros((1, 3))

				y[:] = target_f[0][:]
				y[0][action] = target

				y_train = []
				y_train.append(y.reshape(3,))
				y_train = np.array(y_train)

				self.brain.fit(np.array([[state.A - state.B]]), y_train, epochs=1, verbose=0)


	def act(self, state):
		
		return np.argmax(self.brain.predict(np.array([[state.A - state.B]]))[0]) #0 or 1

	def calc_reward(self, cur_state, next_state, action):

		reward = 0.0

		if (action == 0): #Buy A Sell B
			reward =  cur_state.B - next_state.B + next_state.A - cur_state.A
			self.apos = self.apos + 1
			self.bpos = self.bpos - 1
		elif (action == 1):				#Sell A Buy B
			reward =  cur_state.A - next_state.A + next_state.B - cur_state.B
			self.apos = self.apos - 1
			self.bpos = self.bpos + 1
		else:	#Do nothing
			reward = 0

		return reward

	def run(self, env):

		cur_st = state()
		nxt_st = state()

		fp = open("reward.txt", "w")
		fp.write("time reward\n")


		for num_episodes in range(500):

			self.total_reward = 0.0
			env.reset()
			self.replay(100)
			
			self.apos = 0
			self.bpos = 0

			#Gather the first observation
			cur_st, done, msg = env.step()

			for num_steps in range(steps):
				
				act = self.act(cur_st)

				tmp_st = state()
				tmp_st.A = cur_st.A
				tmp_st.B = cur_st.B

				nxt_st, done, msg = env.step()

				#Calculate reward
				act_reward = self.calc_reward(tmp_st, nxt_st, act)

				self.total_reward = self.total_reward + act_reward

				self.remember(tmp_st, act, act_reward, nxt_st, done)

				cur_st.A = nxt_st.A
				cur_st.B = nxt_st.B

				#Episode over, liquidate everything
				if num_steps == steps - 1:
					self.total_reward = self.total_reward + self.apos * cur_st.A + self.bpos * cur_st.B
					break

			print "Episode: "+str(num_episodes)+" Total Reward: "+str(self.total_reward/1000.0)
			s = str(num_episodes)+" "+str(self.total_reward/1000.0)+"\n"
			fp.write(s)


if __name__ == "__main__":
	np.random.seed(1)
	trade_agent = Agent()
	env = mkt_env()
	Agent.run(trade_agent, env)

