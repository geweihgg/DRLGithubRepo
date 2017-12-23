#!/usr/bin/python

import gym
import argparse

import numpy as np
import tensorflow as tf
import pprint as pp

from gym import wrappers

from simple_actor_critic import Actor, Critic
from ou_noise import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer
from summary import Summary_Udef


def train(sess, env, args, actor, critic, actor_noise):
	summary_udef = Summary_Udef(sess, args['summary_dir'])
	
	sess.run(tf.global_variables_initializer())

	#Initialize the target network.
	actor.update_target_network()
	critic.update_target_network()

	replay_buffer = ReplayBuffer(maxlen=int(args['max_buffer_size']),minlen=int(args['start_buffer_size']), random_seed=int(args['random_seed']))

	total_step = 0

	reward_scale = float(args['reward_scale'])

	for i in range(int(args['max_episodes'])):    #'for loop' for episode

	              #train step
		s = env.reset()

		ep_reward=0        #What's this? This means that the total reward in this episode.
		ep_ave_max_q=0    #What's this?  Sum up max(Q(s,a)) in every step and average over total episode.

		for j in range(int(args['max_step_num'])):
			# if args['render_env']:
			# 	if i % int(args['target_render_interval']) != 0:
			# 		env.render()

			a = actor.predict(np.reshape(s, (1,actor.state_dim))) + actor_noise()

			#For example, input=[[1,2,3]], output=[[0]], so we should use a[0]
			s_, r, terminal, info = env.step(a[0])

			r_scale = reward_scale * r

			replay_buffer.append(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)),  \
				r_scale, terminal, np.reshape(s_, (actor.state_dim,)))

			if replay_buffer.size() > int(args['start_buffer_size']):
				batch_s, batch_a, batch_r, batch_t, batch_s_ = \
				    replay_buffer.sample_batch(int(args['batch_size']))

				#Calculate the target Q batch
				target_q = critic.predict_target(batch_s_, actor.predict_target(batch_s_))

				y_i=[]

				for k in range(int(args['batch_size'])):
					if batch_t[k]:
						y_i.append(batch_r[k])
					else:
						y_i.append(batch_r[k] + critic.gamma * target_q[k])

				y_i_shaped = np.reshape(y_i, (int(args['batch_size']), 1))

				#behavior network predicted Q
				predicted_Q_value, _=critic.train(batch_s, batch_a, y_i_shaped)

				#Use the predicted_q_value to compute the max Q, max over the batch
				#Average is done in summary step
				ep_ave_max_q += np.amax(predicted_Q_value)

				a_outputs=actor.predict(batch_s)
				a_gradient=critic.action_gradients(batch_s, a_outputs)
				actor.train(batch_s, a_gradient[0])

				actor.update_target_network()
				critic.update_target_network()
			s = s_
			ep_reward += r

			if terminal:
				summary_udef.write_summaries_train(ep_reward, ep_ave_max_q/float(j), i)
				print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d}'.format(int(ep_reward), \
					i, (ep_ave_max_q / float(j)), j))
				break

			#record the total_step
			total_step=total_step+1

		print total_step


		#test the target network
		if i % int(args['target_test_interval']) == 0:
			s = env.reset()

			target_network_reward = 0

			for j in range(int(args['max_step_num'])):
				if args['render_env']:
					if i % int(args['target_render_interval']) == 0:
						if j == 0:
							print '-------------------------------------------------'
							print 'target network render without noise'
						env.render()
				a = actor.predict_target(np.reshape(s, (1,actor.state_dim))) 
				s_, r, terminal, info = env.step(a[0])
				target_network_reward += r
				s = s_
				if terminal:
					print('target network reward in test step: ({:d}, {:d})'.format(j, int(target_network_reward)))
					print '-------------------------------------------------'
					summary_udef.write_summaries_test(target_network_reward, i)
					break

			
def main(args):
	with tf.Session() as sess:
		env = gym.make(args['env'])

		np.random.seed(int(args['random_seed']))
		tf.set_random_seed(int(args['random_seed']))
		env.seed(int(args['random_seed']))

		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]

		#Ensure action bound is symmetric
		#Cause this is the minimal feasible version
		# assert(env.action_space.high==-env.action_space.low)
		action_bound = env.action_space.high

		actor = Actor(sess, state_dim, action_dim, action_bound, float(args['actor_lr']), float(args['tau']), float(args['batch_size']))

		critic = Critic(sess, state_dim, action_dim, float(args['critic_lr']), float(args['tau']), \
			float(args['gamma']), actor.get_num_trainable_vars(), float(args['weight_decay']))

		actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
		
		env=wrappers.Monitor(env,args['monitor_dir'])

		train(sess, env, args, actor, critic, actor_noise)



if __name__ == "__main__":

	#File dir
	path='Walker2d/exp3_ddpg_rewardScale'

	parser=argparse.ArgumentParser(description='provide arguments for DDPG agent')

	# agent parameters, default arguments are provided according to the paper
	parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
	parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
	parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
	parser.add_argument('--tau', help='soft target update parameter', default=0.001)
	parser.add_argument('--max-buffer-size', help='max size of the replay buffer', default=1000000)
	parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=64)
	parser.add_argument('--start-buffer-size',help='sample batch start size of replay buffer',default=1000)
	parser.add_argument('--reward-scale',help='scale of reward',default=1)

	#L2 weight decay, weight decay params
	parser.add_argument('--weight-decay',help='weight decay of critic network', default=0.01)

	# run parameters
	parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
	parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
	parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
	parser.add_argument('--max-step-num', help='max step of 1 episode', default=1600)
	parser.add_argument('--target-test-interval', help='interval of rendering target network performance in env', default=1)
	parser.add_argument('--target-render-interval', help='interval of rendering target network performance in env', default=1)
	parser.add_argument('--render-env', help='render the gym env', action='store_true')
	# parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
	parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./monitor/'+path)
	parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./tensorboard/'+path)

	#We use the gym wrappers to save the training records
	parser.set_defaults(render_env=True)
	# parser.set_defaults(use_gym_monitor=True)

	args = vars(parser.parse_args())

	pp.pprint(args)

	main(args)

