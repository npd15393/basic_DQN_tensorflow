import os,sys
import argparse
import tensorflow as tf
import numpy as np
import gym
import random
import time	
import matplotlib.pyplot as plt

#################### Define algorithm parameters ######################
ENV_NAME = 'CartPole-v0'  # Environment name
# FRAME_WIDTH = 84  # Resized frame width
# FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
# STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.9  # Discount factor
EXPLORATION_STEPS = 50000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
# MAX_EP_STEPS = 1000   # maximum time step in one episode
# INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
# FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
# INITIAL_REPLAY_SIZE = 2000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 40000  # Number of replay memory the agent uses for training
BATCH_SIZE = 64  # Mini batch size
TARGET_UPDATE_INTERVAL = 1000  # The frequency with which the target network is updated
# TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
# ACTOR_LEARNING_RATE = 0.0001  # Learning rate used by RMSProp
CRITIC_LEARNING_RATE = 0.001
# MOMENTUM = 0.95  # Momentum used by RMSProp
# MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
# SAVE_INTERVAL = 300000  # The frequency with which the network is saved
# NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
# LOAD_NETWORK = False
# TRAIN = True
# SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
# SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
# NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time
# RENDER=False
TARGET_UPD_FREQ=0.01
EPSILON=0.3

############################ Replay memory class #################################
class ReplayMemory:
	def __init__(self,n):
		self.size=n
		self.expBuffer=[]

	# Circular memory
	def push(self,exp):
		if len(self.expBuffer)<self.size:
			self.expBuffer=[exp]+self.expBuffer
		else:
			self.expBuffer=[exp]+self.expBuffer
			self.expBuffer.pop(-1)

	# Check if buffer has sufficient experience
	def isReady(self,sz):
		return len(self.expBuffer)>=BATCH_SIZE

	def sampleBatch(self,sz):
		idxs=[np.random.randint(0,self.size) for _ in range(sz)]
		return [self.expBuffer[idx] for idx in idxs]

############################# Q Network Definition ################################
class DQN:	
		def __init__(self,nf,na):
			n_l1=24
			self.n_features=nf
			self.n_actions=na
			self.s = tf.placeholder(tf.float32, [None, self.n_features], name="state")
			self.q_targ = tf.placeholder(tf.float32, [None, self.n_actions], "q_val")

			# Network Str: The Deeper the narrower
			# First and last layer Linear activations: I've found it to work better in general
			self.w1=tf.Variable(tf.truncated_normal([self.n_features,n_l1]))
			self.b1=tf.Variable(tf.zeros([n_l1]))

			self.wh=tf.Variable(tf.truncated_normal([n_l1,n_l1/2]))
			self.bh=tf.Variable(tf.zeros([n_l1/2]))

			self.w2=tf.Variable(tf.truncated_normal([n_l1/2,self.n_actions]))
			self.b2=tf.Variable(tf.zeros([self.n_actions]))

			# Define graph computation
			self.out1=tf.add(tf.matmul(self.s,self.w1),self.b1)
			self.outh=tf.nn.relu(tf.add(tf.matmul(self.out1,self.wh),self.bh))
			self.q_est=(tf.add(tf.matmul(self.outh,self.w2),self.b2))

			# Define loss MSE
			self.loss = tf.reduce_mean((self.q_est-self.q_targ)**2)

			self.init=tf.global_variables_initializer()
			
			# Define optimizer SGD
			self.optimizer = tf.train.GradientDescentOptimizer(CRITIC_LEARNING_RATE)#.minimize(self.loss)

			# Clip Gradients
			self.gradients = self.optimizer.compute_gradients(self.loss)

			def ClipIfNotNone(grad):
				if grad is None:
					return grad
				return tf.clip_by_value(grad, -0.1, 0.1)
			self.clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in self.gradients]

			# Apply clipped gradients 
			self.train_q = self.optimizer.apply_gradients(self.clipped_gradients)


################################### Agent Class #####################################
class DQNAgent:
	def __init__(self,env):
		self.n_actions=2
		self.n_features=4
		self.memory=ReplayMemory(NUM_REPLAY_MEMORY)
		tf.reset_default_graph()
		self.mainQN=DQN(4,2)
		self.targQN=DQN(4,2)

	# Policies
	def randomAct(self):
		return np.random.randint(0,self.n_actions)

	def eGreedy(self,st):
		a=random.random()
		if a<EPSILON:
			return random.randint(0,self.n_actions-1)
		else:
			qs=self.sess.run(self.mainQN.q_est,{self.mainQN.s:st.reshape(1,self.n_features)})[0]
			idx=np.argmax(qs)

		return idx


	# Get all target network trainable variables
	def updateTargetGraph(self,tfVars,tau):
		total_vars = len(tfVars)
		op_holder = []
		for idx,var in enumerate(tfVars[0:total_vars//2]):
			op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
		return op_holder

	# Update target network variables
	def updateTarget(self,op_holder):
		for op in op_holder:
			self.sess.run(op)

	# Sample 1 batch from memory; Train; Return current Q function loss
	def learn_from_exp(self):
		exps=self.memory.sampleBatch(BATCH_SIZE)
		Q_targets=[]
		Q_olds=[]
		cur_st=[]
		for exp in exps:
			Q_next=self.sess.run(self.targQN.q_est, {self.targQN.s: exp[2].reshape(1,4)})[0]
			Q_old=self.sess.run(self.mainQN.q_est, {self.mainQN.s: exp[0].reshape(1,4)})[0]
			Q_olds.append(Q_old)

			ac=exp[1]
			if exp[4]:
				Q_old[ac]=exp[3]
			else:
				Q_old[ac]=exp[3]+GAMMA*np.max(Q_next)
			Q_targets.append(Q_old)
			cur_st.append(exp[0])

		cur_st=np.array(cur_st)
		l,_=self.sess.run([self.mainQN.loss,self.mainQN.train_q], {self.mainQN.q_targ:np.array(Q_targets).reshape([BATCH_SIZE,self.n_actions]),\
			self.mainQN.s: cur_st.reshape([BATCH_SIZE,self.n_features])})
		return l

	# Test run; use greedy policy
	def test(self):
		current_state=env.reset()
		done=False
		total_rwd=0
		while not done:
			# env.render()
			qs=self.sess.run(self.mainQN.q_est,{self.mainQN.s:current_state.reshape(1,self.n_features)})[0]
			idx=np.argmax(qs)

			current_state,rew,done,_=env.step(idx)
			total_rwd=total_rwd+rew
		env.reset()
		print("Total Test Reward: "+str(total_rwd))

	def train(self,env):
		cnt=0
		global EPSILON
		current_state=env.reset()

		tfVars=trainables = tf.trainable_variables()
		self.targVars=self.updateTargetGraph(tfVars,0.002)

		self.init = tf.global_variables_initializer()

		with tf.Session() as self.sess:
			self.sess.run(self.init)
			avg_rwd=0
			running_avg=0
			avg_loss=0
			while True:
				if cnt<EXPLORATION_STEPS:
					selected_action=self.randomAct()
				else:
					selected_action=self.eGreedy(current_state)

				next_state, reward, done, _ = env.step(selected_action)
				avg_rwd=avg_rwd+reward
				
				self.memory.push((current_state,selected_action,next_state,reward/10,done),)

				# Reset counters on termination
				if done:
					current_state=env.reset()
					if cnt>EXPLORATION_STEPS:
						avg_rwd=0
				else: 
					current_state=next_state

				if cnt>EXPLORATION_STEPS:
					avg_loss=avg_loss*cnt/(cnt+1)+self.learn_from_exp()/(cnt+1)
					
					# Log
					if cnt%1000==0:
						print("Episode: "+str(cnt)+ " Loss: "+str(avg_loss))
						self.test()
					# if cnt%500==0:
						

					self.updateTarget(self.targVars)
					
					# Reduce exploration 
					EPSILON=EPSILON-1/100

				cnt=cnt+1

		

if __name__=='__main__':
	
	env=gym.make(ENV_NAME)
	agent=DQNAgent(env)
	agent.train(env)