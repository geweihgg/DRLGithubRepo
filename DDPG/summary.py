import tensorflow as tf

class Summary_Udef:
	def __init__(self, sess, summary_dir):
		episode_reward = tf.Variable(0.)
		episode_reward_str = tf.summary.scalar("Reward", episode_reward)
		episode_ave_max_q = tf.Variable(0.)
		episode_ave_max_q_str = tf.summary.scalar("Qmax Value", episode_ave_max_q)
		target_epsode_reward = tf.Variable(0.)
		target_epsode_reward_str = tf.summary.scalar("Target Reward", target_epsode_reward)

		self.sess = sess
		self.summary_vars_str = [episode_reward_str, episode_ave_max_q_str, target_epsode_reward_str]
		self.summary_vars = [episode_reward, episode_ave_max_q, target_epsode_reward]
		self.summary_ops_train = tf.summary.merge([self.summary_vars_str[0], self.summary_vars_str[1]])
		self.summary_ops_test = tf.summary.merge([self.summary_vars_str[2]])

		self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

	def write_summaries_train(self, ep_reward, ep_ave_max_q, i):
		summary_str = self.sess.run(self.summary_ops_train, feed_dict={
			self.summary_vars[0]: ep_reward,
			self.summary_vars[1]: ep_ave_max_q
			})
		self.writer.add_summary(summary_str, i)
		self.writer.flush()

	def write_summaries_test(self, target_epsode_reward, i):
		summary_str = self.sess.run(self.summary_ops_test, feed_dict={
			self.summary_vars[2]: target_epsode_reward
			})
		self.writer.add_summary(summary_str, i)
		self.writer.flush()



