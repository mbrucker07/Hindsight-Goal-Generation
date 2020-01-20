import numpy as np
import time
from common import get_args,experiment_setup
from copy import deepcopy
import pickle

if __name__=='__main__':
	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)

	args.logger.summary_init(agent.graph, agent.sess)

	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)')

	best_success = 0.0

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()
	counter= 0
	for epoch in range(args.epoches):
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()
			start_time = time.time()

			goal_list = learner.learn(args, env, env_test, agent, buffer, write_goals=args.show_goals)
			tester.cycle_summary()

			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epoches))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			args.logger.save_csv()
			"""
			#TODO: new section
			if counter >= 3:
				values = args.logger.save_csv()
				for key, value in values.items():
					if "Success" in key:
						current_success = value
				if current_success >= best_success:
					args.logger.save_agent(agent, "best")
				args.logger.save_agent(agent, "latest")
			counter += 1
			"""
			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)
		if args.learn == 'hgg' and goal_list and args.show_goals != 0:
			name = "{}goals_{}".format(args.logger.my_log_dir, epoch)
			if args.mesh:
				learner.sampler.mesh.plot_graph(goals=goal_list, save_path=name)
			with open('{}.pkl'.format(name), 'wb') as file:
					pickle.dump(goal_list, file)

		tester.epoch_summary()

		#args.logger.save_agent(deepcopy(agent), "periodical") # TODO: New

	tester.final_summary()

