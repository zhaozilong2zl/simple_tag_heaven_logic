import torch
import copy
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, VmasEnv
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator import rendering


from benchmarl.experiment.callback import Callback
class TagAgentScript(Callback):
    """
    TagAgentScript used to freeze the good agents' policy during training.
    """

    def __init__(self):
        super().__init__()

    def on_setup(self):
        print(self.experiment.train_group_map)
        del self.experiment.train_group_map["agent"]


from benchmarl.experiment import ExperimentConfig

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml() # We start by loading the defaults

# Override devices
vmas_device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
experiment_config.sampling_device = vmas_device
experiment_config.train_device = train_device

experiment_config.max_n_frames = 10_000_000 # Number of frames before training ends
experiment_config.gamma = 0.99
experiment_config.on_policy_collected_frames_per_batch = 60_000*3 # Number of frames collected each iteration
experiment_config.on_policy_n_envs_per_worker = 600 # Number of vmas vectorized enviornemnts (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
experiment_config.on_policy_n_minibatch_iters = 45
experiment_config.on_policy_minibatch_size = 4096
experiment_config.evaluation = True
experiment_config.render = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.evaluation_interval = 120_000*3 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
experiment_config.evaluation_episodes = 200 # Number of vmas vectorized enviornemnts used in evaluation
experiment_config.loggers = ["wandb"] # Log to csv, usually you should use wandb

# Loads from "benchmarl/conf/task/vmas/navigation.yaml"
task = VmasTask.SIMPLE_TAG_HEAVEN.get_from_yaml()

# We override the NAVIGATION config with ours
task.config = {
        "max_steps": 300,
        "num_good_agents": 3,
        "num_adversaries": 4,
        "num_landmarks": 2,
        "shared_rew": False,
        "bound" : 2.0,

}

from benchmarl.algorithms import MappoConfig

# We can load from "benchmarl/conf/algorithm/mappo.yaml"
# algorithm_config = MappoConfig.get_from_yaml()

# Or create it from scratch
algorithm_config = MappoConfig(
        share_param_critic=True, # Critic param sharing on
        clip_epsilon=0.2,
        entropy_coef=0.001, # We modify this, default is 0
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
        use_tanh_normal=True,
        minibatch_advantage=False,
    )

from benchmarl.models.mlp import MlpConfig

model_config = MlpConfig(
        num_cells=[256, 256], # Two layers with 256 neurons each
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.Tanh,
    )

# Loads from "benchmarl/conf/model/layers/mlp.yaml" (in this case we use the defaults so it is the same)
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

from benchmarl.experiment import Experiment

experiment_config.max_n_frames = 50_000_000 # Runs one iteration, change to 50_000_000 for full training
# experiment_config.on_policy_n_envs_per_worker = 60 # Remove this line for full training
# experiment_config.on_policy_n_minibatch_iters = 1 # Remove this line for full training

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
    callbacks=[TagAgentScript()],
)
experiment.run()




























