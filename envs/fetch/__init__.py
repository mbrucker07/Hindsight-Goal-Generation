from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .mesh import MeshGoalEnv
from .no_mesh import NoMeshGoalEnv

# TODO: change this file for new env handling!
def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv,
		'mesh': MeshGoalEnv,
		'no_mesh': NoMeshGoalEnv,
	}[args.goal](args)