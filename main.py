from q_learning import *
from q_networks.vanilla_dqn import VanillaDQN
from q_networks.dueling_dqn import DuelingDQN
from replay.random_replay import RandomReplay

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

#----------------------------------------------------
# Double DQN use can be enabled using this
double_dqn = True

# for experience replay to be randomly sampled
replay_method = RandomReplay(obs_dim, memory_size, batch_size)

# Using the simplest DQN
network = VanillaDQN(obs_dim, 128, action_dim)

# Using Dueling DQN
network = DuelingDQN(obs_dim, 128, action_dim)
#-----------------------------------------------------

# Initilizing the agent
agent = DQNAgent(env, network, replay_method, memory_size, batch_size, target_update, epsilon_decay, double_dqn)

# Training the agent
agent.train(num_frames)

# Tests the agent
agent.test()