import cv2
import gym
import random
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # --- ICM components ---
        self.feature_size = 512
        self.feature = nn.Sequential(
            self.cnn_base,
            nn.Linear(self.cnn_output_size, self.feature_size),
        )
        
        self.forward_net = nn.Sequential(
            nn.Linear(n_actions + self.feature_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.feature_size)
        )
        
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_actions)
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
    
    def icm_predict(self, state, next_state, action):
        state_feature = self.feature(state)
        next_state_feature = self.feature(next_state)

        # get pred action
        pred_action = torch.cat((state_feature, next_state_feature), 1)
        pred_action = self.inverse_net(pred_action)

        # get pred next state
        pred_next_state_feature = torch.cat((state_feature, action), 1)
        pred_next_state_feature = self.forward_net(pred_next_state_feature)

        return next_state_feature, pred_next_state_feature, pred_action


class Agent(object):
    def __init__(self):
        # Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.action_space = gym.spaces.Discrete(12)
        input_channels, height, width, action_size = 4, 84, 84, 12
        self.action_size = action_size
        self.input_channels = input_channels
        self.height = height
        self.width = width

        self.q_net = ConvQNet(input_channels, height, width, action_size).to(device)
        self.q_net.load_state_dict(torch.load("icm_model.pth", map_location=device))
        self.q_net.eval()

        self._obs_buffer = np.zeros((2,) + (240, 256, 3), dtype=np.uint8)
        self.frames = deque([], maxlen=4)
        self.frame_skip_count = 0
        self.last_action = 0

    def get_action(self, state, deterministic=True):
        # Implement the action selection
        if deterministic:
            with torch.no_grad():
                state_np = np.array(state) # list -> array -> tensor
                return self.q_net(state_np).max(1).indices.item()
        else:
            # Boltzmann Exploration
            with torch.no_grad():
                state_np = np.array(state) # list -> array -> tensor
                q_values = self.q_net(state_np) / 0.5  # a high tau means more randomness
                probabilities = F.softmax(q_values, dim=1)
                action = torch.multinomial(probabilities, num_samples=1).item()
                return action
            # return random.randint(0, self.action_size - 1)
    
    def format_observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # (240, 256)
        observation = cv2.resize(
            observation, (84, 84), interpolation=cv2.INTER_AREA
        ) # (84, 84)
        observation = np.array(observation).astype(np.float32) / 255.0
        self.frames.append(observation) # (4, 84, 84)
        return np.expand_dims(np.array(list(self.frames)), 0)  # (1, 4, 84, 84)
    
    def MaxAndSkipEnvAct(self, observation):
        if len(self.frames) < 3:
            stacked_observation = self.format_observation(observation)
            return self.last_action
    
        if self.frame_skip_count > 0:
            if self.frame_skip_count == 1:
                self._obs_buffer[0] = observation
        else:
            self.frame_skip_count = 4
            self._obs_buffer[1] = observation
            max_frame = self._obs_buffer.max(axis=0)
            stacked_observation = self.format_observation(max_frame)
            # self.last_action = self.get_action(stacked_observation, deterministic=True)
            if random.random() < 0.1:
                self.last_action = self.get_action(stacked_observation, deterministic=False)
            else:
                self.last_action = self.get_action(stacked_observation)

        self.frame_skip_count -= 1
        return self.last_action

    def act(self, observation):
        return self.MaxAndSkipEnvAct(observation)