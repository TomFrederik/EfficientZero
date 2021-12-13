import math
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize


def MLP(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        num_obs_channels,
        num_layers,
        num_hidden,
        num_channels,
        momentum=0.1,
    ):
        """Representation network
        Parameters
        ----------
        num_obs_channels: int
            number of observation channels
        num_layers: int
            number of hidden layers
        num_hidden: int
            number of hidden units per layer
        num_channels: int
            size of the latent state
        momentum: float
            momentum for batch normalization
        """
        super().__init__()
        modules = [nn.Linear(num_obs_channels, num_hidden), nn.BatchNorm1d(num_hidden, momentum=momentum), nn.ReLU(inplace=True)]
        for i in range(num_layers - 1):
            modules.extend([nn.Linear(num_hidden, num_hidden), nn.BatchNorm1d(num_hidden, momentum=momentum), nn.ReLU(inplace=True)])
        modules.extend([nn.Linear(num_hidden, num_channels)])
        self.net = nn.Sequential(*modules)
        
    def forward(self, x):
        # print(f'{x.shape = }')
        B, T, D = x.shape
        x = x.reshape((B, T*D))
        # print(f'{x.shape = }')
        x = self.net(x)
        # print(f'{x.shape = }')
        # x = x.reshape((B, T, -1))
        # print(f'{x.shape = }')
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        num_channels,
        fc_reward_layers,
        full_support_size,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward MLP
        """
        super().__init__()
        self.num_channels = num_channels
        
        self.lstm_hidden_size = lstm_hidden_size
        # print(f'lstm input size = {self.num_channels}')
        # print(f'{self.lstm_hidden_size = }')
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = MLP(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:,:-1]
        x = x.unsqueeze(0)

        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        # dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        # for block in self.resblocks:
        #     for name, param in block.named_parameters():
        #         dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        # dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        # return dynamic_mean
        return 0

    def get_reward_mean(self):
        # reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        # for name, param in self.fc.named_parameters():
        #     temp_weights = param.detach().cpu().numpy().reshape(-1)
        #     reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        # reward_mean = np.abs(reward_w_dist).mean()
        # return reward_w_dist, reward_mean
        return 0, 0


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        action_space_size,
        num_channels,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_channels: int
            channels of hidden states
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        init_zero: bool
            True -> zero initialization for the last layer of value/policy MLP
        """
        super().__init__()
        # maybe use 1d Batchnorm before passing input to MLPs?
        # print(f"{num_channels = }")
        self.fc_value = MLP(num_channels, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = MLP(num_channels, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        x = x.squeeze()
        # print(f"{x.shape = }")
        # print(self.fc_policy)
        # print(self.fc_value)
        # B, T, D = x.shape
        # x = x.reshape((B, T*D))
        # print(f"{x.shape = }")
        value = self.fc_value(x)
        policy = self.fc_policy(x)
        # print(f"{value.shape = }")
        # print(f"{policy.shape = }")
        # value = value.reshape((B, T, -1))
        # policy = policy.reshape((B, T, -1))
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        num_obs_channels,
        action_space_size,
        num_channels,
        num_layers,
        num_hidden,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        bn_mt=0.1,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False
    ):
        """EfficientZero network
        Parameters
        ----------
        num_obs_channels: int
            number of observations
        action_space_size: int
            action space
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy MLP
        state_norm: bool
            True -> normalization for hidden states
        """
        super().__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size

        self.representation_network = RepresentationNetwork(
            num_obs_channels,
            num_layers,
            num_hidden,
            num_channels,
            momentum=bn_mt,
        )

        self.dynamics_network = DynamicsNetwork(
            num_channels + 1,
            fc_reward_layers,
            reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_channels,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        self.projection_in_dim = num_channels
        self.projection = nn.Sequential(
            nn.Linear(self.projection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        observation = observation.squeeze()
        # print(f'{observation.shape = }')
        encoded_state = self.representation_network(observation)
        # print(f'{encoded_state = }')
        if not self.state_norm:
            # print('not normalizing!')
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            # print(f'{encoded_state_normalized = }')
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                )
            )
            .to(action.device)
            .float()
        )
        # print(f'{action.shape = }')
        # print(f'{action_one_hot.shape = }')
        action_one_hot = (
            action * action_one_hot / self.action_space_size
        )
        # print(f'{action_one_hot.shape = }')
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        # print(f'{x.shape = }')
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)
        # print(f'{next_encoded_state.shape = }')
        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        # hidden_state = hidden_state.view(-1, self.projection_in_dim) # not necessary in this case
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

