import matplotlib.pyplot as plt
import numpy as np


class Plotter():
	def __init__(self,history,session_length):
		self.history=history
		self.session_length=session_length




	def average_ranks(self):
		counter=0
		ranks=np.zeros((1+self.session_length))

			 	
		for user in self.history.keys():
			for gt in self.history[user]:
				while len(self.history[user][gt])<1+self.session_length:
					self.history[user][gt].append(self.history[user][gt][-1])
				ranks= ranks+np.array((self.history[user][gt]))
				counter+= 1
		return ranks/counter




	def plot_ranks(self):
		plt.figure()
		average_ranks=self.average_ranks()
		sessions=np.arange(1+self.session_length)
		plt.plot(sessions,average_ranks,'r-o')
		plt.xlabel("Step")
		plt.ylabel("Average Ground Truth Rank")
		plt.savefig("test.png")


