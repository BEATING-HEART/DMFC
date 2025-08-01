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

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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

    def __init__(
        self, env, T=10, grid_h=4, grid_w=4, scale_factor=0.01, json_file=None
    ):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.json_file = json_file

        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        # avail = torch.tensor(avail).view(1, 1, R).float()
        # future_inflows = torch.tensor(future_inflows).view(1, self.T, R).float()
        # future_demands = torch.tensor(future_demands).view(1, self.T, R).float()
        avail = (
            torch.tensor([obs[0][n][self.env.time + 1] for n in self.env.region])
            .view(1, 1, self.env.nregion)
            .float()
        )
        future_inflows = (
            torch.tensor(
                [
                    [
                        (obs[0][n][self.env.time + 1] + self.env.dacc[n][t])
                        for n in self.env.region
                    ]
                    for t in range(self.env.time + 1, self.env.time + self.T + 1)
                ]
            )
            .view(1, self.T, self.env.nregion)
            .float()
        )

        # demand_input = np.zeros((self.T, self.env.nregion, self.env.nregion))
        # price = np.zeros((self.T, self.env.nregion, self.env.nregion))
        # for t in range(self.T):
        #     for (i, j), d in self.env.scenario.demand_input.items():
        #         demand_input[t, i, j] = d[t + self.env.time + 1]
        #     for (i, j), p in self.env.price.items():
        #         price[t, i, j] = p[t + self.env.time + 1]
        # fd_alt = np.sum(demand_input * price, axis=2)
        future_demands = (
            torch.tensor(
                [
                    [
                        sum(
                            [
                                (self.env.scenario.demand_input[i, j][t])
                                * (self.env.price[i, j][t])
                                if (i, j) in self.env.scenario.G.edges
                                else 0
                                for j in self.env.region
                            ]
                        )
                        for i in self.env.region
                    ]
                    for t in range(self.env.time + 1, self.env.time + self.T + 1)
                ]
            )
            .view(1, self.T, self.env.nregion)
            .float()
        )

        x = (
            torch.cat(
                (avail, future_inflows, future_demands),
                dim=1,
            )
            .squeeze(0)
            .view(21, self.env.nregion)
            .T
        )
        x *= self.s
        if self.json_file is not None:
            edge_index = torch.vstack(
                (
                    torch.tensor(
                        [edge["i"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                    torch.tensor(
                        [edge["j"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                )
            ).long()
        else:
            adjacency = self.env.scenario.adjacency
            rows, cols = np.where(adjacency == 1)
            edge_index = torch.tensor(np.vstack((rows, cols)))

            # gird case
            # edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        # log.warning(f"x.sum: {x.sum()}")
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
        env,
        input_size,
        json_file=None,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
    ):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        self.json_file = json_file

        self.actor = GNNActor(self.input_size, self.hidden_size)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = GNNParser(self.env, json_file=self.json_file)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def forward(self, obs, jitter=1e-20):
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

    def select_action(self, obs):
        concentration, value = self.forward(obs)
        # log.warning(f"concentration: {concentration}")

        m = Dirichlet(concentration)

        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), value))
        return action.cpu().numpy()
        # return list(action.cpu().numpy())

    def training_step(self):
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
        self.optimizers["a_optimizer"].step()

        self.optimizers["c_optimizer"].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
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
