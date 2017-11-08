#!/usr/bin/python

import tensorflow as tf
import tflearn


###############################  Actor  ####################################

class Actor(object):
	"""
	This class defines a neural network approximator for  policy.
	And it's under a deterministic policy.
	Attention: the outputs of actor network is assumed to be symmetric,
	        [-action_bound,action_bound] 
	we can rewrite the output layer to make the network compatible with other problems.
	"""
	def __init__(self,sess,state_dim,action_dim,action_bound,learning_rate,tau,BATCH_SIZE):
		"""
		Initializer.
		state_dim: the input dim of the network.
		action_dim: the output dim of the network.
		action_bound: the bound used for clipping the output of the network.
		tau: the ratio for update the target networks.
		BATCH_SIZE: used for computing the average of gradients over the batch.
		"""
		self.sess=sess
		self.state_dim=state_dim
		self.action_dim=action_dim
		self.action_bound=action_bound
		self.learning_rate=learning_rate
		self.tau=tau

		#behavior network
		self.inputs,self.outputs,self.scaled_out=self.create_actor_network()
		self.network_params=tf.trainable_variables()

		#target network
		self.target_inputs,self.target_outputs,self.target_scaled_out=self.create_actor_network()
		self.target_network_params=tf.trainable_variables()[len(self.network_params):]

		#Op for updating target network
		self.soft_update=[tf.assign(t,(self.tau*b+(1-self.tau)*t)) \
		        for b,t in zip(self.network_params,self.target_network_params)]

		#This gradient will be provided by the  critic network(behavior network).
		self.action_gradient=tf.placeholder(tf.float32,[None,self.action_dim])

		#Use the gradient provided by the cirtic network to weight the gradients of actor network(behavior network).
		#action_gradient: [None, self.action_dim]
		#scaled_out: [None, self.action_dim]
		#network_params: All trainable variables in behavior actor network, for example, 3.
		#actor_gradients: [[None,self.action_dim], [None,self.action_dim], [None,self.action_dim]]
		#TODO: We should divide self.actor_gradients by N, or divide the learning rate by N.
		self.actor_gradients=tf.gradients( \
			self.scaled_out,self.network_params,self.action_gradient)

		#Optimization Op
		#-self.learning_rate for ascent policy
		opt=tf.train.AdamOptimizer(-self.learning_rate/BATCH_SIZE)
		self.optimize=opt.apply_gradients(zip(self.actor_gradients,self.network_params))

		self.num_trainable_vars=len(self.network_params)+len(self.target_network_params)

	def create_actor_network(self):
		"""
		Create a simple network for low dimensional inputs.
		Two fully connected hidden layers.
		"""
		#Input layer
		inputs=tflearn.input_data(shape=[None,self.state_dim])
		net=tflearn.layers.normalization.batch_normalization(inputs)

		#Fc1
		w_init1=tflearn.initializations.uniform(minval=-tf.div(1.0,tf.sqrt(float(self.state_dim))),maxval=tf.div(1.0,tf.sqrt(float(self.state_dim))))
		net=tflearn.fully_connected(net,400,weights_init=w_init1)
		net=tflearn.layers.normalization.batch_normalization(net)
		net=tflearn.activations.relu(net)

		#Fc2
		w_init2=tflearn.initializations.uniform(minval=-tf.div(1.0,tf.sqrt(float(400))),maxval=tf.div(1.0,tf.sqrt(float(400))))
		net=tflearn.fully_connected(net,300,weights_init=w_init2)
		net=tflearn.layers.normalization.batch_normalization(net)
		net=tflearn.activations.relu(net)

		#output layer
		#As in the paper, final layer weights are initialized to Uniform[-3e-3,3e-3]
		w_init=tflearn.initializations.uniform(minval=-3e-3,maxval=3e-3)
		outputs=tflearn.fully_connected(net,self.action_dim,activation='tanh')    #'tanh'-->(-1,1)

		#Scale output to -action_bound to action_bound
		scaled_out=tf.multiply(outputs,self.action_bound)
		return inputs,outputs,scaled_out

	def train(self,inputs,a_gradient):
		self.sess.run(self.optimize,feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def predict(self,inputs):
		return self.sess.run(self.scaled_out,feed_dict={
			self.inputs: inputs
		})

	def predict_target(self,inputs):
		return self.sess.run(self.target_scaled_out,feed_dict={
			self.target_inputs: inputs
		})

	def update_target_network(self):
		self.sess.run(self.soft_update)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars



###############################  Critic  ####################################

class Critic(object):
	"""
	This class defines a neural network for value function.
	The action must be obtained from the output of the actor network.
	"""
	def __init__(self,sess,state_dim,action_dim,learning_rate,tau,gamma,num_actor_vars,weight_decay):
		"""
		Initializer.
		tau: the ratio for update the target networks.
		gamma: the factor used in the computation of target y_i.
		weight decay: the weight decay params.
		"""
		self.sess=sess
		self.state_dim=state_dim
		self.action_dim=action_dim
		self.learning_rate=learning_rate
		self.tau=tau
		self.gamma=gamma
		self.weight_decay=weight_decay

		#behavior network
		self.inputs,self.actions,self.outputs=self.create_critic_network()
		self.network_params=tf.trainable_variables()[num_actor_vars:]

		#target network
		self.target_inputs,self.target_actions,self.target_outputs=self.create_critic_network()
		self.target_network_params=tf.trainable_variables()[(len(self.network_params)+num_actor_vars):]

		self.soft_update=[tf.assign(t,(self.tau*b+(1-self.tau)*t)) \
		        for b,t in zip(self.network_params,self.target_network_params)]

		#Network target(y_i)
		self.predicted_q_value=tf.placeholder(tf.float32,shape=[None,1])

		#Define loss and optimization Op
		#Adam algorithm.
		#This can automatically update the changeable variables,
		#thus there is no need to use apply_gradients.
		self.loss=tflearn.mean_square(self.predicted_q_value,self.outputs)
		self.optimize=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		#Get the gradient of the net w.r.t. the action.
		
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		# This means that when we use this function to compute gradients wrt. 
		# the input, it won't sum up over the batch.
		#This is not the same case as in the actor's gradients.
		self.action_gradient=tf.gradients(self.outputs,self.actions)

	def create_critic_network(self):
		"""
		Create a neural network for low dimensional inputs.
		"""
		#input layer
		inputs=tflearn.input_data(shape=[None,self.state_dim])
		actions=tflearn.input_data(shape=[None,self.action_dim])
		net=tflearn.layers.normalization.batch_normalization(inputs)

		#Fc1
		w_init1=tflearn.initializations.uniform(minval=-tf.div(1.0,tf.sqrt(float(self.state_dim))),maxval=tf.div(1.0,tf.sqrt(float(self.state_dim))))
		net=tflearn.fully_connected(net,400,weights_init=w_init1,regularizer='L2',weight_decay=self.weight_decay)
		net=tflearn.layers.normalization.batch_normalization(net)
		net=tflearn.activations.relu(net)

		#Fc2
		#Add the action tensor in the 2nd hidden layer
		#Use two temp layers to get the corresponding weights and biases
		w_init2=tflearn.initializations.uniform(minval=-tf.div(1.0,tf.sqrt(float(400))),maxval=tf.div(1.0,tf.sqrt(float(400))))
		t1=tflearn.fully_connected(net,300,weights_init=w_init2,regularizer='L2',weight_decay=self.weight_decay)
		w_init3=tflearn.initializations.uniform(minval=-tf.div(1.0,tf.sqrt(float(self.action_dim))),maxval=tf.div(1.0,tf.sqrt(float(self.action_dim))))
		t2=tflearn.fully_connected(actions,300,weights_init=w_init3,regularizer='L2',weight_decay=self.weight_decay)
		net=tflearn.activations.relu(tf.matmul(net,t1.W)+tf.matmul(actions,t2.W))

		#As in the paper, final layer weights are initialized to Uniform[-3e-3,3e-3]
		w_init=tflearn.initializations.uniform(minval=-3e-3,maxval=3e-3)
		outputs=tflearn.fully_connected(net,1,weights_init=w_init,regularizer='L2',weight_decay=self.weight_decay)
		return inputs,actions,outputs

	def train(self,inputs,actions,predicted_q_value):
		return self.sess.run([self.outputs,self.optimize],feed_dict={
			self.inputs: inputs,
			self.actions: actions,
			self.predicted_q_value: predicted_q_value    #predicted_q_value means y_i, it's the target q value
		})

	def predict(self,inputs,actions):
		return self.sess.run(self.outputs,feed_dict={
			self.inputs: inputs,
			self.actions: actions
		})

	def predict_target(self,inputs,actions):
		return self.sess.run(self.target_outputs,feed_dict={
			self.target_inputs: inputs,
			self.target_actions: actions
		})

	def update_target_network(self):
		self.sess.run(self.soft_update)

	def action_gradients(self,inputs,actions):
		return self.sess.run(self.action_gradient,feed_dict={
			self.inputs: inputs,
			self.actions: actions
		})
