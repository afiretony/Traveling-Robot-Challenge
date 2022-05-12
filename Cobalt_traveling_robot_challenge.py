import numpy as np
from math import dist, sqrt
import matplotlib.pyplot as plt
from functools import lru_cache
import heapq
from scipy.spatial import distance_matrix

"""
## Planning: The Traveling Robot Problem

Visit a collection of points in the shortest path you can find. 
The catch? You have to "go home to recharge" every so often. 

We want fast approximations rather than a brute force perfect solution.
Your solution will be judged on:
* the length of path it produces
* fast runtime
* code quality and maintainability

### Details

* There are 5000 points distributed uniformly in [0, 1]
* The recharge station is located at (.5, .5)
* You cannot travel more than 3 units of distance before recharging
* You must start and end at the recharge station
* Skeleton code provided in Python. Python and C++ are acceptable
"""
class Traveling_robot:

	def __init__(self, N=5000):
		#############################
		self.home = np.array([0.5, 0.5]) # home is the recharging station
		self.max_charge = 3.0
		#############################

		# generate the points to visit uniformly in [0,1]
		# recharging station is index 0
		self.N = N
		np.random.seed(0)
		self.pts = np.vstack((self.home, np.random.rand(self.N,2)))
		self.order = []
		

	def check_order(self):
		"""Check whether a given order of points is valid, and prints the total 
		length. You start and stop at the charging station.
		pts: np array of points to visit, prepended by the location of home
		order: array of pt indicies to visit, where 0 is home
		i.e. order = [0, 1, 0, 2, 0, 3, 0]"""

		print("Checking order")
		assert(self.pts.shape == (self.N + 1, 2)) # nothing weird
		assert(self.order[0] == 0) # start path at home
		assert(self.order[-1] == 0) # end path at home
		assert(set(self.order) == set(range(self.N + 1))) # all self.pts visited

		print("Assertions passed")

		# traverse path
		total_d = 0
		charge = self.max_charge
		last = self.pts[0,:]

		for idx in self.order:
			pt = self.pts[idx, :]
			d = np.linalg.norm(pt - last)
			
			# update totals
			total_d += d
			charge -= d

			assert(charge > 0) # out of battery

			# did we recharge?
			if idx == 0:
				charge = self.max_charge

			# moving to next point
			last = pt

		# We made it to end! path was valid
		print("Valid path!")
		print(total_d)

	def draw_path(self):
		"""Draw the path to the screen"""
		path = self.pts[self.order, :]

		plt.plot(path[:,0], path[:,1])
		plt.show()

	def path_finder_naive(self):
		'''
		Naive approach to generate the order, 
		smallest time complexity but not optimal for path at all
		'''
		self.order.append(0)
		for i in range(self.N):
			self.order.append(i+1)
			self.order.append(0)

	def compute_distance_matrix(self):
		'''
		Computes a distance matrix where dist[i][j] indicates distance from point i to j
		'''
		# naive way
		# self.dist = np.zeros((self.N+1, self.N+1))
		# for i in tqdm.tqdm(range(self.N+1)):
		# 	for j in range(self.N+1):
		# 		if j > i:
		# 			self.dist[i][j] = np.linalg.norm(self.pts[i] - self.pts[j])
		# 			self.dist[j][i] = self.dist[i][j]

		self.dist = distance_matrix(self.pts, self.pts)
		

	def path_finder_exact(self):
		'''
		Find exact solution of traveling salesman problem using memorization,
		and seperate path to segments that is robot-reachable/returnable
		'''
		# exact solution
		self.compute_distance_matrix()
		visited = frozenset([0])

		# step 1: Finding optimal solution of traveling salesman
		# DFS 
		@lru_cache(maxsize=None)
		def DFS(curr, visited):
			if len(visited) == self.N+1:
				return (self.dist[curr][0], [curr])
			else:
				# enumerating all possible next point
				res = []
				for i in range(1, self.N+1):
					if i not in visited:
						# deep copy visited list and append next point
						ls_visited = list(visited)
						ls_visited.append(i)
						visited_copy = frozenset(ls_visited)

						d, order = DFS(i, visited_copy)

						# using heapq to always get the minimum distance path
						heapq.heappush(
							res,
							( d + self.dist[curr][i], order)
						)

				distance, order = heapq.heappop(res) 
				neworder = order.copy()
				neworder.append(curr)
				return (distance, neworder)
				
		_, opt_order = DFS(0, visited)
		
		# step 2: segment exact solution to sub paths
		self.order = [0, opt_order[0]]
		d = self.dist[0][opt_order[0]]
		for i in range(self.N):
			cur_pt = opt_order[i]
			next_pt = opt_order[i+1]
			# check if homing needed
			if d + self.dist[cur_pt][next_pt]+ self.dist[next_pt][0] > self.max_charge:
				# return home
				self.order.append(0)
				d = self.dist[0][next_pt]
			else:
				d += self.dist[cur_pt][next_pt]
			self.order.append(next_pt)
		print(self.order)
	
	def path_finder_NN(self):
		'''
		implementation of nearest neighbour search algorithm
		'''
		self.compute_distance_matrix()
		print('Distance map created')

		visited = set()
		curr = 0 # current node index
		d = 0
		self.order.append(0)
		
		while visited != set(range(self.N + 1)):
			# search for nearset neighbour
			min_dist = 2.0 # initialize, must larger distance between two points
			visited.add(curr)
			for i in range(self.N+1):
				if i not in visited:
					if self.dist[curr][i] < min_dist:
						min_idx = i
						min_dist = self.dist[curr][i]

			if d + min_dist + self.dist[min_idx][0] < self.max_charge:
				# go to the next node
				curr = min_idx
				self.order.append(min_idx)
				d += min_dist
			else:
				# need charging, next state is charged and continue from origin
				d = 0.0
				self.order.append(0)
				curr = 0

		self.order.append(0)


if __name__ == "__main__":
	import timeit
	start = timeit.default_timer()
	print('--------------function stdout-------------')
	robot = Traveling_robot(5000)
	# robot.path_finder_naive()
	# robot.path_finder_exact() # may take forever to compute with 5000 points, make sure select less points (~10) to check the implementation
	robot.path_finder_NN() # working model
	robot.check_order()
	print('\n--------------------end-------------------')
	stop = timeit.default_timer()
	print('\n Time: ', stop - start) 
	robot.draw_path()
	
	
