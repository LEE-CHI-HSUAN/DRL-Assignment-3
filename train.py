# # Environment Setup

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
print("action space     :", env.action_space.n)
print("observation space:", env.observation_space.shape)

import gym
import numpy as np
from collections import deque
from gym import spaces
from gym.wrappers import ResizeObservation, GrayScaleObservation

# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        ret = np.array(observation).astype(np.float32) / 255.0
        ret = np.transpose(ret, (2, 0, 1))
        return np.expand_dims(ret, 0)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-3)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def mario_warpper(env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)
    return env


env = mario_warpper(env)
env.observation_space

# # DQN Agent

import torch
from torch import nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class ConvQNet(nn.Module):
    def __init__(
        self, input_channels, height, width, n_actions
    ):  # Smaller LSTM might work too
        super(ConvQNet, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.n_actions = n_actions

        self.cnn_base = nn.Sequential(
            # Input: (Batch*SeqLen, C, H, W) e.g. (Batch*SeqLen, 1, 84, 84) if grayscale
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), # (Batch*SeqLen, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),             # (Batch*SeqLen, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),             # (Batch*SeqLen, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output for the next layer
        )

        # Calculate the flattened CNN output size dynamically
        self.cnn_output_size = self._get_cnn_output_size()

        # --- Dueling Streams ---
        # Input size is now lstm_hidden_size
        self.value_stream = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),  # Can adjust hidden sizes here
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),  # Can adjust hidden sizes here
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_cnn_output_size(self):
        """Helper function to calculate the CNN output size."""
        with torch.no_grad():
            # Create a dummy input matching the observation space
            dummy_input = torch.zeros(1, self.input_channels, self.height, self.width)
            output = self.cnn_base(dummy_input)
            return output.shape[1]  # Return the flattened size

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, device=device, dtype=torch.float32)

        x = self.cnn_base(x)

        # --- Process through Dueling Streams ---
        value = self.value_stream(x)  # (batch_size, seq_len, 1)
        advantage = self.advantage_stream(x)  # (batch_size, seq_len, n_actions)
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize the buffer
        self.memory = deque(maxlen=capacity)

    # Implement the add method
    def add(self, state, action, reward, next_state, done):
        # Add an experience to the buffer
        self.memory.append((state, action, reward, next_state, done))

    # Implement the sample method
    def sample(self, size):
        if len(self.memory) < size:
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, size)


class DQNVariant:
    def __init__(self, input_channels, height, width, action_size):
        # Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.action_size = action_size
        self.input_channels = input_channels
        self.height = height
        self.width = width

        self.q_net = ConvQNet(input_channels, height, width, action_size).to(device)
        self.target_net = ConvQNet(input_channels, height, width, action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=5e-4)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.TAU = 1e-3
        self.gamma = 0.95
        self.batch_size = 64

    def add(self, state, action, reward, next_state, done):
        # Add an experience to the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def get_action(self, state, deterministic=True):
        # Implement the action selection
        if deterministic:
            with torch.no_grad():
                state_np = np.array(state) # list -> array -> tensor
                return self.q_net(state_np).max(1).indices.item()
        else:
            return random.randint(0, self.action_size - 1)

    def update(self):
        # Implement hard update or soft update
        for q_param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.TAU * q_param.data + (1 - self.TAU) * target_param.data)

    def train(self):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)  # Improvement 3
        # seperate batch into state, action, reward, next_state, done
        states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
        states_b = np.concatenate(states_b)
        actions_b = torch.tensor(actions_b, device=device).unsqueeze(1)
        rewards_b = torch.tensor(rewards_b, device=device).unsqueeze(1)
        next_states_b = np.concatenate(next_states_b)
        dones_b = torch.tensor(dones_b, device=device, dtype=torch.float32).unsqueeze(1)

        # Compute TD-target
        with torch.no_grad():
            # get the best next action if not done
            next_actions_b = self.q_net(next_states_b).max(1).indices
            # get the state-action values from target net
            next_q_values = self.target_net(next_states_b).gather(1, next_actions_b.unsqueeze(1))
            target_q_values = rewards_b + self.gamma * next_q_values * (1 - dones_b)

        # Compute loss and update the model
        q_values = self.q_net(states_b).gather(1, actions_b)
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()  # Improvement 3
        self.optimizer.step()


# ## Train

from itertools import count

agent = DQNVariant(4, 84, 84, 12)
model_name = input("model name: ")
if model_name != "":
    print("Loading...", end="")
    agent.q_net.load_state_dict(torch.load(model_name, map_location=device))
    agent.target_net.load_state_dict(torch.load(model_name, map_location=device))
    print("done!")
else:
    print("Train a model from scratch.")

# Hyperparameters
num_episodes = 100000
epsilon_start = 1
epsilon_end = 0.1
epsilon_decay = 0.997
epsilon = epsilon_start
train_freq = 1
update_freq = 5
global_steps = 0

reward_history = []  # Store the total rewards for each episode
max_score = 0
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    far = 0
    # epsilon = max(epsilon * epsilon_decay, epsilon_end)
    epsilon = 0.1

    for t in count():
        # action selection
        # epsilon = np.cos(global_steps / 1000) * 0.09 + 0.1
        if random.random() < epsilon:
            action = agent.get_action(state, deterministic=False)
        else:
            action = agent.get_action(state)

        # Take a step in the environment
        next_state, reward, done, info = env.step(action)

        # Add the experience to the replay buffer and train the agent
        agent.add(state, action, reward, next_state, done)
        global_steps += 1
        if global_steps % train_freq == 0:
            agent.train()
        if global_steps % update_freq == 0:
            agent.update()

        # Update the state and total reward
        state = next_state
        total_reward += reward

        if done:
            break
        # print(f"\r[{t}] eps= {epsilon:.2f}, x: {info['x_pos']}, y: {info['y_pos']}", end='   ')
        far = max(far, info["x_pos"])

    print(
        f"\rEpisode {episode + 1}, tot Reward: {total_reward}, farest {far}",
        " " * 20,
        end="   ",
    )

    reward_history.append(total_reward)
    avg_score = np.mean(reward_history[-100:])
    if avg_score > max_score:
        max_score = avg_score
        torch.save(agent.q_net.state_dict(), "MarioAgent.pth")

    if (episode + 1) % 100 == 0:
        print(
            f"\rEpisode {episode + 1}, Avg Reward: {avg_score:7.2f}, farest {far}",
            " " * 20,
        )
        torch.save(agent.q_net.state_dict(), f"mario_net/ckpt-{episode + 1}.pth")

from datetime import datetime

torch.save(agent.q_net.state_dict(), f"mario_net/dqn-{str(datetime.now())}.pth")
