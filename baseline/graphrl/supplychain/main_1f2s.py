from __future__ import print_function

import argparse
import json
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import tomli
import torch
from loguru import logger
from tqdm import trange

sys.path.append(os.getcwd())
from src.algos.graph_rl_agent import A2C, Actor, Critic, GNNParser
from src.algos.lcp_solver import solveLCP
from src.algos.mpc import MPC
from src.algos.stype_policy import s_type_policy
from src.envs.scim_env import Network, SupplyChainIventoryManagement

parser = argparse.ArgumentParser(description="A2C-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=1010, metavar="S", help="random seed (default: 10)"
)

# Model parameters
parser.add_argument(
    "--algo",
    type=str,
    default="rl",
    help='defines the algorithm to evaluate (only "rl" can use --test=False)',
)
parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="run test mode (default: False)",
)
parser.add_argument(
    "--directory",
    type=str,
    default="saved_files",
    help="defines directory where to save files",
)
parser.add_argument(
    "--log_level",
    type=int,
    default=0,
    help="specify a log level",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=20000,
    metavar="N",
    help="number of episodes to train agent (default: 20k)",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=30,
    metavar="N",
    help="number of steps per episode (default: T=60)",
)
parser.add_argument("--no-cuda", type=bool, default=True, help="disables CUDA training")
parser.add_argument(
    "--use_wandb",
    action="store_true",
    default=False,
    help="use wandb for logging",
)
parser.add_argument(
    "--s_store",
    type=int,
    default=16,
    metavar="S",
    help="optimal store order-up-to-level (default: 16)",
)
parser.add_argument(
    "--s_factory",
    type=int,
    default=19,
    metavar="S",
    help="optimal factory order-up-to-level (default: 19)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.seed > 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## loguru init
logger.remove()
format_string = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{file}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
logger.add(
    sys.stderr,
    level=args.log_level,
    format=format_string,
    colorize=True,
    backtrace=True,
    diagnose=True,
)
# logger.add(
#     Path(__file__).parent / f"{args.dataset}.log",
#     level=args.log_level,
#     format=format_string,
# )

# with open(Path(__file__).parent / "config.toml", "rb") as f:
with open(Path(__file__).parents[3] / "config.toml", "rb") as f:
    data = tomli.load(f)
setattr(args, "cplexpath", data.get("ibm_cplex"))

# Define SCIM Simulator Environment
G = nx.DiGraph()
edge_list = [(0, 1), (0, 2)]
G.add_edges_from(edge_list)
for e in G.edges:
    G.edges[e]["capacity"] = 1000

# Set network parameters
dmax = [2, 16]
dvar = [2, 2]
tf = 30
factory_nodes = [0]
warehouse_nodes = [1, 2]
production_time = 1
product_prices = [15]
production_costs = [5]
storage_capacities = [20, 9, 12]
storage_costs = [3, 2, 1]
edge_costs = [0.3, 0.6]
edge_time = [1, 1]
network = Network(
    G=G,
    tf=tf,
    dmax=dmax,
    dvar=dvar,
    factory_nodes=factory_nodes,
    warehouse_nodes=warehouse_nodes,
    product_prices=product_prices,
    production_costs=production_costs,
    storage_capacities=storage_capacities,
    storage_costs=storage_costs,
    randomize_graph_args=(None, "random-tt"),
    randomize_demand_args=(None, "single-od"),
    edge_costs=edge_costs,
    edge_time=edge_time,
)
env = SupplyChainIventoryManagement(network)

if args.algo == "rl":
    # Initialize agent
    parser = GNNParser(env, edge_list=edge_list)
    actor = Actor(
        node_size=10, edge_size=2, hidden_dim=64, out_channels=1, num_factories=1
    )
    critic = Critic(node_size=10, edge_size=2, hidden_dim=64, out_channels=1)
    model = A2C(
        env,
        parser,
        actor,
        critic,
        clip=5,
        baseline=None,
        parametrization="Gaus-Dirichlet",
    )

if not args.test:
    #######################################
    #############Training Loop#############
    #######################################

    # Initialize lists for logging
    log = {"train_reward": [], "train_served_demand": [], "train_reb_cost": []}
    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode

    for i_episode in epochs:
        obs = env.reset()  # initialize environment
        episode_reward = 0
        for step in range(T):
            # use Graph-RL policy (RL)
            (prod, ship), (gaus_log_prob, dir_log_prob) = model.select_action(
                obs, show_log_prob=True
            )
            # solve LCP
            action = solveLCP(
                env, ship, prod, CPLEXPATH=args.cplexpath, res_path="scim_1f2s"
            )
            # Take action in environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            # Store the transition in memory
            model.rewards.append(reward)
            # stop episode if terminating conditions are met
            if done:
                break
        # perform on-policy backprop
        grad_norms = model.training_step()

        # Send current statistics to screen
        epochs.set_description(
            f"Episode {i_episode + 1} | Reward: {episode_reward:.2f} | Grad Norms: Actor={grad_norms['a_grad_norm']:.2f}, Critic={grad_norms['v_grad_norm']:.2f}"
        )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(path=f"./saved_files/ckpt/1f2s/graph_rl.pth")
            best_reward = episode_reward
        # Log KPIs
        log["train_reward"].append(episode_reward)
        model.log(log, path=f"./{args.directory}/rl_logs/1f2s/graph_rl.pth")
else:
    np.random.seed(args.seed)
    if args.algo == "rl":
        # Load pre-trained model
        model.load_checkpoint(
            path=f"{Path(__file__).parent}/{args.directory}/pretrained/1f2s/graph_rl.pth"
        )
    test_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(test_episodes)  # epoch iterator
    # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}
    task_reward_list = []
    for episode in epochs:
        episode_reward = 0
        obs = env.reset()
        done = False
        k = 0
        while not done:
            if args.algo == "oracle":
                action = MPC(env.time, env, T=10, CPLEXPATH=args.cplexpath)
            else:
                if args.algo == "rl":
                    with torch.no_grad():
                        a_probs, value = model(obs)
                        mu, sigma = a_probs[0][0], a_probs[0][1]
                        alpha = a_probs[1]
                    prod, ship = mu, alpha / (alpha.sum() + 1e-16)
                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    action = solveLCP(
                        env, ship, prod, CPLEXPATH=args.cplexpath, res_path="scim_1f2s"
                    )
                if args.algo == "heur":
                    prod, ship = s_type_policy(env, args.s_store, args.s_factory)
                    action = (prod, ship)
                if args.algo == "random":
                    avg_demand = torch.tensor(
                        np.mean(
                            [
                                sum([env.demand[t][n] for n in range(len(list(env.G)))])
                                for t in range(env.tf)
                            ]
                        )
                    ).view(
                        1,
                    )
                    prod, ship = (
                        avg_demand,
                        np.random.dirichlet(np.ones(len(list(env.G)) - 1)),
                    )
                    action = solveLCP(
                        env, ship, prod, CPLEXPATH=args.cplexpath, res_path="scim_1f2s"
                    )
            # Take action in environment
            obs, rebreward, done, info = env.step(action)
            episode_reward += rebreward
            # track performance over episode
            k += 1
        task_reward_list.append(episode_reward)
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode + 1} | Reward: {episode_reward:.2f} | Aggregated: {np.mean(task_reward_list):.0f} +- {np.std(task_reward_list):.0f}"
        )
