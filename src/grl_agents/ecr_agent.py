"""
A2C-GNN
-------
This file contains the Graph-RL agent specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import json
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid

import wandb

if TYPE_CHECKING:
    from src.envs import ECRObs

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])
args = namedtuple("args", ("render", "gamma", "log_interval"))
args.render = True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################


class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, T=10, scale_factor=0.01):
        super().__init__()
        self.T = T
        self.s = scale_factor

    def parse_obs(self, state: "ECRObs"):
        timestamp = state["timestamp"]
        avail = state["available_cars"]
        R = len(avail)

        next_step = timestamp + 1
        future_inflows = np.sum(state["flows"], axis=(1, 2))[
            # summation over types and origin nodes
            next_step + 1 : next_step + self.T + 1
        ]
        future_inflows += avail

        future_demands = (
            state["graph"].plam[next_step : next_step + self.T]
            * state["graph"].adj_matrix
        )
        price = state["graph"].price[next_step : next_step + self.T]
        future_demands = np.sum(future_demands * price, axis=2)

        avail = torch.tensor(avail).view(1, 1, R).float()
        future_inflows = torch.tensor(future_inflows).view(1, self.T, R).float()
        future_demands = torch.tensor(future_demands).view(1, self.T, R).float()

        x = (avail, future_inflows, future_demands)
        x = torch.cat(x, dim=1).squeeze(0).view(21, R).T
        x = x * self.s
        rows, cols = np.where(state["graph"].adj_matrix == 1)
        edge_index = torch.vstack((torch.tensor(rows), torch.tensor(cols))).long()
        data = Data(x, edge_index)
        return data


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)

    #     self.init_weights()

    # def init_weights(self):
    #     torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
    #     torch.nn.init.kaiming_uniform_(self.lin1.weight)
    #     torch.nn.init.kaiming_uniform_(self.lin2.weight)
    #     torch.nn.init.kaiming_uniform_(self.lin3.weight)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## CRITIC ###################
#########################################


class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)

    #     self.init_weights()

    # def init_weights(self):
    #     torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
    #     torch.nn.init.kaiming_uniform_(self.lin1.weight)
    #     torch.nn.init.kaiming_uniform_(self.lin2.weight)
    #     torch.nn.init.kaiming_uniform_(self.lin3.weight)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## A2C AGENT ################
#########################################


class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        input_size,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
    ):
        super(A2C, self).__init__()
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device

        self.actor = GNNActor(self.input_size, self.hidden_size)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = GNNParser()

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def forward(self, obs: "ECRObs", jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out = self.actor(x)
        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, value

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, obs: "ECRObs"):
        concentration, value = self.forward(obs)

        m = Dirichlet(concentration)

        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), value))
        return action

    def training_step(self, t):
        use_wandb = False
        if wandb.run is not None:
            use_wandb = True

        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(
                F.smooth_l1_loss(value, torch.tensor([R]).to(self.device))
            )

        # take gradient steps
        self.optimizers["a_optimizer"].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        if use_wandb:
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    wandb.log(
                        {
                            "actor_loss": torch.stack(policy_losses).sum().item(),
                            f"actor_grad_{name}_norm": param.grad.norm().item(),
                        },
                        step=t,
                    )
        self.optimizers["a_optimizer"].step()

        self.optimizers["c_optimizer"].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        if use_wandb:
            for name, param in self.critic.named_parameters():
                if param.grad is not None:
                    wandb.log(
                        {
                            "critic_loss": torch.stack(value_losses).sum().item(),
                            f"critic_grad_{name}_norm": param.grad.norm().item(),
                        },
                        step=t,
                    )
        self.optimizers["c_optimizer"].step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=1e-3)
        optimizers["c_optimizer"] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
