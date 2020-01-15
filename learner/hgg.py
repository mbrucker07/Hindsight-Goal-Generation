import copy
import numpy as np
from envs import make_env
from envs.utils import goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
from envs.distance_mesh import DistanceMesh

#TODO: replaced goal_distance with get_mesh_goal_distance

class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		if self.args.mesh:
			self.create_mesh_distance()

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.get_mesh_goal_distance(obs['achieved_goal'],obs['desired_goal']) #TODO: added self.get_mesh_
			if dis>self.max_dis: self.max_dis = dis

	def create_mesh_distance(self): #TODO: new
		obstacles = list()
		push_region = [1.3, 0.75, 0.6, 0.25, 0.35, 0.2]
		push_obstacles = [[1.3 - 0.125, 0.75, 0.6 - 0.18, 0.125, 0.04, 0.1]]
		mesh = DistanceMesh(region=push_region, spaces=[50, 50, 2], obstacles=push_obstacles)
		mesh.compute_cs_graph()
		mesh.compute_dist_matrix()
		self.mesh = mesh

	def get_mesh_goal_distance(self, goal_a, goal_b): #TODO: new
		#print("HGG goals: {} // {}".format(goal_a, goal_b))
		if self.args.mesh:
			#print("{} vs. {}".format(self.mesh.get_dist(goal_a, goal_b), np.linalg.norm(goal_a - goal_b, ord=2)))
			return self.mesh.get_dist(goal_a, goal_b)
		else:
			return 	np.linalg.norm(goal_a - goal_b, ord=2)

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
		if noise_std is None: noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx):
		if self.args.env[:5]=='Fetch':
			return self.add_noise(self.pool[idx])
		else:
			return self.pool[idx].copy()

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		candidate_goals = []
		candidate_edges = []
		candidate_id = []

		agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
			feed_dict = {
				agent.raw_obs_ph: obs
			}
			value = agent.sess.run(agent.q_pi, feed_dict)[:,0]
			value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
			achieved_value.append(value.copy())

		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):
				res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma))
				match_dis = np.min(res)+self.get_mesh_goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c # TODO: added self.get_mesh_
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		assert match_count==self.length

		explore_goals = [0]*self.length
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals)==self.length
		self.pool = np.array(explore_goals)

class HGGLearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)

		self.env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))

		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
		self.sampler = MatchSampler(args, self.achieved_trajectory_pool)

	def learn(self, args, env, env_test, agent, buffer):
		initial_goals = []
		desired_goals = []
		for i in range(args.episodes):
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy()
			initial_goals.append(goal_a.copy())
			desired_goals.append(goal_d.copy())

		self.sampler.update(initial_goals, desired_goals)

		achieved_trajectories = []
		achieved_init_states = []
		for i in range(args.episodes):
			obs = self.env_List[i].get_obs()
			init_state = obs['observation'].copy()
			explore_goal = self.sampler.sample(i)
			self.env_List[i].goal = explore_goal.copy()
			obs = self.env_List[i].get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			achieved_trajectories.append(np.array(trajectory))
			achieved_init_states.append(init_state)
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			if self.sampler.get_mesh_goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01: # TODO: added self.sampler.get_mesh_
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())