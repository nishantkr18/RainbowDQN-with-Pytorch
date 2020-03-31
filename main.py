from q_learning import *
from q_networks.vanilla_dqn import VanillaDQN
from q_networks.dueling_dqn import DuelingDQN
from q_networks.noisy_dqn import NoisyDQN
from q_networks.categorical_dqn import CategoricalDQN
from replay.random_replay import RandomReplay
from replay.prioritized_replay import PrioritizedReplay

# environment
env_id = "CartPole-v0"
env = gym.make(env_id)

# parameters
num_frames = 2000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Double DQN use can be enabled using this
double_dqn = True

# # for experience replay to be randomly sampled
# replay_method = RandomReplay(obs_dim, memory_size, batch_size)

#----------------------------------------------------
# # using N-Step Learning
# n_step = 3
# gamma = 0.99
# memory = RandomReplay(obs_dim, memory_size, batch_size, n_step = n_step)
# memory_n = RandomReplay(obs_dim, memory_size, batch_size, n_step = n_step, gamma = gamma)
# network = VanillaDQN(obs_dim, 128, action_dim)
# agent = DQNAgent(env, network, memory, memory_size, batch_size, target_update, epsilon_decay, double_dqn, n_step=n_step, gamma=gamma, memory_n=memory_n)

# # Using the simplest DQN
# network = VanillaDQN(obs_dim, 128, action_dim)
# agent = DQNAgent(env, network, replay_method, memory_size, batch_size, target_update, epsilon_decay, double_dqn, n_step=1)

# # Use Prioritized Experience Replay with vanilla DQN
# alpha = 0.2
# replay_method = PrioritizedReplay(obs_dim, memory_size, batch_size, alpha)
# network = VanillaDQN(obs_dim, 128, action_dim)
# agent = DQNAgent(env, network, replay_method, memory_size, batch_size, target_update, epsilon_decay, double_dqn, n_step=1, PER=True)

# # Using Dueling DQN
# network = DuelingDQN(obs_dim, 128, action_dim)
# agent = DQNAgent(env, network, replay_method, memory_size, batch_size, target_update, epsilon_decay, double_dqn, n_step=1)

# # Using Noisy DQN
# network = NoisyDQN(obs_dim, 128, action_dim)
# agent = DQNAgent(env, network, replay_method, 
# 				memory_size, batch_size, target_update, 
# 				epsilon_decay, double_dqn=True, n_step=1, 
# 				is_noisy=True) # is_noisy creates two conditions: resets noise after each update & removes epsilon greedy from action_selection

# # using Categorical DQN
# v_min = 0.0
# v_max = 200.0
# atom_size = 51
# support = torch.linspace(v_min, v_max, atom_size).to(device)
# network = CategoricalDQN(obs_dim, 128, action_dim, atom_size, support) 
# agent = DQNAgent(env, network, replay_method, 
# 				memory_size, batch_size, target_update, 
# 				epsilon_decay, double_dqn=True, is_noisy=False, n_step=1,
# 				is_categorical=True, v_min=v_min, v_max=v_max, # is_categorical enables the special loss function calculation for categorical DQN
# 				atom_size=atom_size)

# using Rainbow DQN
n_step = 3
gamma = 0.99
v_min = 0.0
v_max = 200.0
atom_size = 51
memory = PrioritizedReplay(obs_dim, memory_size, batch_size, n_step = n_step)
memory_n = PrioritizedReplay(obs_dim, memory_size, batch_size, n_step = n_step, gamma = gamma)
support = torch.linspace(v_min, v_max, atom_size).to(device)
network = RainbowDQN(obs_dim, 128, action_dim, atom_size, support) 
agent = DQNAgent(env, network, memory, 
				memory_size, batch_size, target_update, 
				epsilon_decay, double_dqn=True, is_noisy=True, n_step=n_step, gamma=gamma, memory_n=memory_n,
				is_categorical=True, v_min=v_min, v_max=v_max, # is_categorical enables the special loss function calculation for categorical DQN
				atom_size=atom_size)

#-----------------------------------------------------

# Training the agent
agent.train(num_frames)

# Tests the agent
agent.test()