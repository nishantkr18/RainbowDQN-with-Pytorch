Rainbow: Combining Improvements in Deep Reinforcement 
=====================================================

The repository is structured in a way that all the different extensions can be turned on/off independently.
This would provide: 
1) A better way to benchmark the use of different extensions
2) Maximum code reusability.

All the extensions requiring network changes are present in q_networks folder:
- **VanillaDQN** - Contains the simple DQN feed forward network
- **Noisy** - Contains the Noisy Layer added
- **Dueling** - Contains the Advantage and value streams added to vanillaDQN
- **Categorical** - contains the distributional element
- **Rainbow** - Contains all the combined network for **Vanilla + Noisy + Categorical**

The rest i.e. N-step, PER and Double can be enabled/disabled by appropiate methods explained below.

#### DQN:
For simple DQN, agent may be initialized as:
```
network = VanillaDQN(obs_dim, 128, action_dim)
agent = DQNAgent(env, network, replay_method, memory_size, batch_size, target_update, epsilon_decay, double_dqn, n_step=1)
```

#### Double DQN:
For double DQN, the parameter `double_DQN` in `DQNAgent` should be set `True`


#### Prioritised Experience Replay:
The replay method needs to be initialized
```
alpha = 0.2
replay_method = PrioritizedReplay(obs_dim, memory_size, batch_size, alpha)

```
and PER parameter in `DQNAgent` needs to be `True`.


#### Dueling Network Architecture 
Since Dueling only involves change in network structure, it uses a special class `DuelingDQN` in `q_networks`
`network = DuelingDQN(obs_dim, 128, action_dim)`

#### Noisy Nets
Again, a change in network type. So `NoisyDQN` to be used, with `is_noisy` set `True`.


#### Multi-step Returns 
Two kings of memory are used. Example given in main.py.

#### Distributional/Categorical RL
Again requires a different network structure, so can be used using `CategoricalDQN` class.

-----------------------------------------
# Rainbow
The `Rainbow` class contains the combination of Categorical, Noisy and Dueling networks.
The implementation contains all the extensions activated.
Can be implemented using:
```
#----------------------------------------------------
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
```


References
==========

[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  