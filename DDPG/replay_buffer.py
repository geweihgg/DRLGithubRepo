#!/usr/bin/python

from collections import deque
from collections import namedtuple
import logging
import random
import numpy as np

class ReplayBuffer(object):
	def __init__(self, maxlen, minlen=0, random_seed=123):
		"""
		Initializer.
		'minlen' means that we should collect enough data before training the model.
		'maxlen' means the maximum buffer size.
		"""
		self.minlen = minlen
		self.maxlen = maxlen
		self.count = 0
		self.buffer = deque()
		self.dtype = namedtuple('Transition',['s','a','r','t','s_'])    #Data type for the replay buffer, t-->done

		#Set the random seed, thus we can have different permutations of the transition.
		random.seed(random_seed)

	def data(self):
		"""
		return the data in the buffer.
		"""
		return self.buffer

	def size(self):
		return self.count

	def append(self,s,a,r,t,s_):
		experience = self.dtype(s,a,r,t,s_)
		if self.count < self.maxlen:
			self.buffer.append(experience)
			self.count += 1
		else:
			logging.debug('Memory bank is full ({})'.format(self.maxlen))
			self.buffer.popleft()
			self.buffer.append(experience)

	def sample_batch(self,batch_size):    #Use 'pop' as the func name?
		"""
		batch_size specifies the number of experience to add
		to the batch. If the replay buffer has less than batch_size
		elements, throw out exception.
		"""
		batch = []

		assert self.count >= self.minlen, 'Replay buffer dose not have enough experiences!'
		assert self.count >= batch_size, 'The total buffer size should be greater than batch size.'

		batch = random.sample(self.buffer, batch_size)

		s = np.array([_.s for _ in batch])
		a = np.array([_.a for _ in batch])
		r = np.array([_.r for _ in batch])
		t = np.array([_.t for _ in batch])
		s_ = np.array([_.s_ for _ in batch])

		return s, a, r, t, s_

	def clear(self):
		self.buffer.clear()
		self.count = 0

def main():
	replay_buffer = ReplayBuffer(minlen=0, maxlen=1e6)
	s1 = 1
	a = 0
	r = 10
	t = 0    #Assume 0 means done
	s2 = 2
	replay_buffer.append(s1, a, r, t, s2)
	#The output would be: deque([Transitions(s=1,a=0,r=10,t=0,s_=2)])
	print replay_buffer.data() 

	print replay_buffer.sample_batch(1)

if __name__ == "__main__":
	main()

