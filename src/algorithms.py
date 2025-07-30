from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from loguru import logger as log

# from baseline.graphrl.mobility.src.algos.lcp_solver import solve as solveECR
# from baseline.graphrl.supplychain.src.algos.lcp_solver import solve as solveSCIM
from src.cplex.solver import ecr_lcp_solver, scim_lcp_solver
from src.envs import CarStatusEnum
from src.grl_agents.ecr_agent import A2C as ECRA2C
from src.grl_agents.scim_agent import A2C as SCIMA2C
from src.utils import fraction_allocation, load_config, random_rounding

if TYPE_CHECKING:
    from src.envs import ECRObs, SCIMObs
    from src.graphs import ECRGraph, SCIMGraph


class Algorithm(ABC):
    @abstractmethod
    def execute(self):
        pass


class SCIMDMFC(Algorithm):
    def __init__(self, mode: str = "test"):
        assert mode == "test", "only test mode is supported"
        self.algo_name = "SCIM-DMFC"
        self.mode = mode

    def execute(
        self,
        graph: "SCIMGraph",
        current_inventory: np.ndarray,
        inventory_capacity: np.ndarray,
        historical_demands: np.ndarray,
        pending_demand: np.ndarray,
        resource_inflow: np.ndarray = None,
        storage_costs: np.ndarray = None,
        transportation_costs: np.ndarray = None,
        manufacturing_cost: int = None,
        penalty_cost: int = None,
        selling_price: int = None,
        demand_decay: float = 0.95,
        **kwargs,
    ):
        """
        Note:
            current_inventory (np.ndarray): inventory level at time [t] when the algorithm is called
            historical_demands (np.ndarray):  historical demand for [t - n1 + 1, ..., t]
            pending_demand (np.ndarray): demand that has not been fulfilled at time [t]
            resource_inflow (np.ndarray): inflow of resources at time [t + 1, ..., t + n2]
        """

        lookahead_d = len(historical_demands)
        weights = np.power(demand_decay, np.flip(np.arange(lookahead_d)))
        wavg_historical_demand = np.average(historical_demands, axis=0, weights=weights)
        pred_next_demand = wavg_historical_demand + pending_demand

        tmp = np.concatenate((historical_demands, wavg_historical_demand[None, :]))
        pred_nnext_demand = np.average(tmp[1:], axis=0, weights=weights)
        pred_nnext_demand = pred_nnext_demand + pending_demand

        current_demand = historical_demands[-1]  # lambda_t
        equilibrium_inventory = (  # s_star
            np.maximum(wavg_historical_demand, current_demand) + pending_demand
        )
        equilibrium_inventory[0] = equilibrium_inventory[1:].sum()

        prod_lb = wavg_historical_demand.sum() / 2
        prod_ub = max(
            0,
            inventory_capacity.sum() + pred_next_demand.sum() - current_inventory.sum(),
        )
        prod_at_least = min(prod_ub, prod_lb)

        # ## fluid scaling
        # N = 1  # normalization factor
        # assert N != 0
        s = current_inventory
        s_star = equilibrium_inventory
        s_ub = inventory_capacity
        lam = pred_next_demand
        prod_at_least = prod_at_least

        log.debug(f"s: {current_inventory}")
        log.debug(f"equili: {equilibrium_inventory}")
        log.debug(f"capa: {inventory_capacity}")
        log.debug(f"nextdemand: {pred_next_demand}")
        log.debug(prod_at_least)

        ## solve
        m = gp.Model("scim-dmfc")
        m.setParam("OutputFlag", 0)
        m.setParam("FeasibilityTol", 1e-7)

        f = m.addVars(graph.n_edges, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="f")
        a = m.addVars(graph.n_nodes, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="a")
        s_prime = m.addVars(
            graph.n_nodes, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="s_prime"
        )
        prod = m.addVars(
            graph.n_nodes, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="prod"
        )
        k = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="k")

        ## overstock penalty
        s_ub_eps = m.addVars(
            graph.n_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="s_ub_eps"
        )
        penalty = gp.quicksum(s_ub_eps[n] for n in range(graph.n_nodes))

        # l1 distance
        dis = m.addVars(
            graph.n_nodes, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="dis"
        )
        diff = m.addVars(
            graph.n_nodes, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="diff"
        )
        m.addConstrs(
            (diff[n] == s_star[n] - s_prime[n] * k for n in range(graph.n_nodes))
        )
        m.addConstrs((dis[n] == gp.abs_(diff[n]) for n in range(graph.n_nodes)))

        # kl divergence
        # div = m.addVars(graph.n_nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="div")
        # log_div = m.addVars(
        #     graph.n_nodes, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="log_div"
        # )
        # m.addConstrs(
        #     (div[n] * s_prime[n] == s_star[n] for n in range(graph.n_nodes))
        # )  # div = s_star / s_prime
        # py = np.concatenate((np.arange(-10, 1, 2), np.arange(1, 5)))
        # px = np.exp(py)
        # for n in range(graph.n_nodes):
        #     m.addGenConstrPWL(div[n], log_div[n], px, py, "pwl_log")
        # m.addConstrs(
        #     # normalization: gp.quicksum(s_prime[n] for n in range(graph.n_nodes))
        #     (
        #         dis[n] * gp.quicksum(s_prime[n] for n in range(graph.n_nodes))
        #         == s_star[n] * log_div[n]
        #         # dis[n] == s_star[n] * log_div[n]
        #         for n in range(graph.n_nodes)
        #     ),
        #     "dis",
        # )

        ## minimax objective (availability)
        t = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="t")
        m.update()

        obj = t - 1e6 * penalty
        obj -= gp.quicksum(dis[n] for n in range(graph.n_nodes))
        m.setObjective(obj, GRB.MAXIMIZE)

        m.addConstrs((t <= a[i] for i in graph.store_nodes), "c0")
        m.addConstrs(  # state transition for factory nodes
            (
                s[n]
                - k * gp.quicksum(f[e_id] for e_id in graph.out_edges_indices(n))
                + k * prod[n]
                == s_prime[n] * k
                for n in graph.factory_nodes
            ),
            "c1-f",
        )
        m.addConstr(gp.quicksum(s_prime[n] for n in range(graph.n_nodes)) == 1)
        m.addConstrs(  # state transition for store nodes
            (
                s[n] + k * gp.quicksum(f[e_id] for e_id in graph.in_edges_indices(n))
                == s_prime[n] * k
                for n in graph.store_nodes
            ),
            "c1-s",
        )
        m.addConstr(  # factory production bounds
            (
                k * gp.quicksum(prod[n] for n in graph.factory_nodes) >= prod_at_least
                # (gp.quicksum(prod[n] for n in graph.factory_nodes) == prod_at_least)
            ),
            "c2-prod",
        )
        for n in graph.factory_nodes:
            if s[n] > 0:
                # only if there is available inventory
                m.addConstr(  # factory shipment bounds
                    (
                        k * gp.quicksum(f[e_id] for e_id in graph.out_edges_indices(n))
                        <= s[n]
                    ),
                    f"c2-ship-f{n}",
                )
            elif s[n] == 0:
                jitter = 1e-20
                m.addConstrs((f[e_id] == jitter for e_id in graph.out_edges_indices(n)))
                # m.addConstrs((f[e_id] == 0 for e_id in graph.out_edges_indices(n)))

        m.addConstrs(  # factory storage bounds
            (
                k * s_prime[n]
                - k * gp.quicksum(f[e_id] for e_id in graph.out_edges_indices(n))
                <= s_ub[n] + s_ub_eps[n]
                for n in graph.factory_nodes
            ),
            "c3-f",
        )
        m.addConstrs(  # store storage bound
            (
                k * s_prime[n] - a[n] * lam[n] <= s_ub[n] + s_ub_eps[n]
                for n in graph.store_nodes
            ),
            "c3-s",
        )

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            log.debug(f"m.Status: {m.Status}")
            m.computeIIS()
            m.write("model1.ilp")
            raise RuntimeError("model is infeasible")
        log.debug(f"objvalue: {m.getObjective().getValue()}")

        prod = np.array([prod[i].x for i in range(graph.n_nodes)])
        f = np.array([f[i].x for i in range(graph.n_edges)])
        s_prime = np.array([s_prime[i].x for i in range(graph.n_nodes)])
        k = k.x

        log.debug(f"s_prime: {s_prime}")
        log.debug(f"k: {k}")

        assert np.isclose(current_inventory.sum() + k * prod.sum(), k * s_prime.sum())

        ## get action
        prod = {i: random_rounding(prod[i] * k) for i in graph.factory_nodes}
        ship = {}
        for n in graph.factory_nodes:
            if s[n] == 0:
                # no available inventory to schedule.
                policy = ...  # no shipment
                continue

            out_edge_indices = graph.out_edges_indices(n)
            stay = s[n] / k - f[out_edge_indices].sum()

            out_flow = np.append(f[out_edge_indices], stay)
            out_edge_indices = np.append(out_edge_indices, -1)
            policy = out_flow / out_flow.sum()

            ### sample from policy
            # counter = Counter(
            #     np.random.choice(out_edge_indices, size=current_inventory[n], p=policy)
            # )
            counter = fraction_allocation(
                current_inventory[n], len(out_edge_indices), policy, out_edge_indices
            )

            counter.pop(-1, None)  # remove stay
            ship.update(dict(counter))

        ship = {graph.edge_list[k]: v for k, v in ship.items()}
        # log.warning(f"prod: {prod}")
        # log.warning(f"ship: {ship}")
        log.debug("===========================")
        # raise NotImplementedError
        return prod, ship


class SCIMGRL(Algorithm):
    def __init__(
        self,
        mode: str = "test",
        dataset: str = "1f2s",
        ckpt_path: str | Path = None,
        use_cuda: bool = False,
    ):
        self.algo_name = "SCIM-GRL"
        self.dataset = dataset
        self.mode = mode
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self._model = SCIMA2C(device=device).to(device)
        self._ckptpath = ckpt_path
        if self.mode == "train":
            self._model.train()
        elif self.mode in ["test", "equivalence_check"]:
            self._model.load_checkpoint(path=self._ckptpath)
            self._model.eval()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        log.info(f"ckpt path: {self._ckptpath}")

    def execute(
        self,
        graph: "SCIMGraph",
        current_inventory: np.ndarray,
        inventory_capacity: np.ndarray,
        historical_demands: np.ndarray,
        pending_demand: np.ndarray,
        resource_inflow: np.ndarray = None,
        storage_costs: np.ndarray = None,
        transportation_costs: np.ndarray = None,
        manufacturing_cost: int = None,
        penalty_cost: int = None,
        selling_price: int = None,
        demand_decay: float = 0.95,
        **kwargs,
    ):
        obs: "SCIMObs" = kwargs.get("obs", None)

        if self.mode == "train":
            (prod, ship), (gaus_log_prob, dir_log_prob) = self._model.select_action(
                obs, show_log_prob=True
            )
        elif self.mode in ["test", "equivalence_check"]:
            with torch.no_grad():
                a_probs, value = self._model(obs)
                mu, sigma = a_probs[0][0], a_probs[0][1]
                alpha = a_probs[1]
            prod, ship = mu, alpha / (alpha.sum() + 1e-16)
            # log.warning(f"rl_prod: {prod}")
            # log.warning(f"rl_ship: {ship}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.mode in ["test", "equivalence_check"]:
            lcp_data = obs["lcp_data"]
            availableProd = [
                (i, max(lcp_data["lcp_state"][i], 0)) for i in graph.factory_nodes
            ]
            desiredShip = [
                (s_id, int(ship[idx] * sum([v for _, v in availableProd])))
                for idx, s_id in enumerate(graph.store_nodes)
            ]
            desiredProd = [
                (i, max(int(prod[i].item()), 0)) for i in graph.factory_nodes
            ]
            storageCapacity = [(i, inventory_capacity[i]) for i in graph.node_list]
            # warehouseStock = [(i, current_inventory[i]) for i in graph.store_nodes]
            warehouseStock = [(i, lcp_data["lcp_state"][i]) for i in graph.store_nodes]
            edge_time = graph.get_edge_attributes("edge_time")
            edge_cost = graph.get_edge_attributes("edge_cost")
            edgeAttr = [
                (i, j, edge_time[i, j], edge_cost[i, j]) for i, j in graph.edge_list
            ]
            demand = [(i, obs["historical_demands"][-1][i]) for i in graph.store_nodes]
            path = load_config(
                Path(__file__).parent.parent / "config.toml", "ibm_cplex"
            )
            action = scim_lcp_solver(
                t=obs["timestamp"],
                availableProd=availableProd,
                desiredShip=desiredShip,
                desiredProd=desiredProd,
                storageCapacity=storageCapacity,
                warehouseStock=warehouseStock,
                edgeAttr=edgeAttr,
                demand=demand,
                edges=graph.edge_list,
                factories=graph.factory_nodes,
                res_path=f"scim_{self.dataset}",
                CPLEXPATH=path,
                # env, ship, prod, CPLEXPATH=args.cplexpath, res_path="scim_1f2s"
            )
            prod, ship = action
            # log.warning(f"prod: {prod}")
            # log.warning(f"ship: {ship}")
            return prod, ship


class ECRDMFC(Algorithm):
    def __init__(self, mode: str = "test"):
        assert mode == "test", "only test mode is supported"
        self.algo_name = "ECR-DMFC"
        self.mode = mode

        self.demand_decay = 0.95
        self.d_ahead_steps = 10

    def execute(self, obs: "ECRObs"):
        graph = obs["graph"]
        empty_cars = obs["available_cars"]
        to_arrive = self._state_lookahead(
            cur_state=empty_cars,
            future_flows=obs["future_flows"],
            avg_demand_time=graph.avg_demand_time,
            demand_mask=graph.demand_mask,
            reb_mask=graph.reb_mask,
            tt_f=graph.tt_f,
            tt_e=graph.tt_e,
        )

        # mean-field scaling
        surrogate_state = empty_cars + to_arrive
        M = np.sum(surrogate_state)
        state = surrogate_state / M

        target_state = self._demand_lookahead(
            state=state,
            historical_demands=obs["historical_demands"],
            d_ahead=self.d_ahead_steps,
        )

        R = graph.n_nodes
        travel_time = graph.tt_e
        reb_mask = graph.reb_mask

        # return {} # for testing
        m = gp.Model("ecr-dmfc")
        m.setParam("OutputFlag", 0)
        m.setParam("FeasibilityTol", 1e-7)

        actual_state = m.addVars(
            R, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="actual_state"
        )
        flow = m.addVars(R, R, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="e")
        dis = m.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="dis")

        m.setObjective(
            gp.quicksum(
                flow[i, j] * reb_mask[i, j] * travel_time[i, j]
                for i in range(R)
                for j in range(R)
                if i != j
            )
            + 1e6 * gp.quicksum(dis[i] for i in range(R)),
            GRB.MINIMIZE,
        )

        m.addConstrs(
            (
                gp.quicksum(
                    flow[j, i] * reb_mask[j, i] - flow[i, j] * reb_mask[i, j]
                    for j in range(R)
                    if j != i
                )
                + state[i]
                == actual_state[i]
                for i in range(R)
            ),
            "c0",
        )
        m.addConstrs(
            (
                gp.quicksum(flow[i, j] * reb_mask[i, j] for j in range(R) if j != i)
                <= state[i]
                for i in range(R)
            ),
            "c1",
        )
        m.addConstrs(
            (actual_state[i] + dis[i] >= target_state[i] for i in range(R)), "c2"
        )
        # m.addConstrs((dis[n] == gp.abs_(eps[n]) for n in range(R)), "c4")

        # define bi KL divergence
        # div = m.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="div")
        # m.addConstrs((div[i] * s_star[i] == s_prime[i] for i in range(R)), "c2")
        # m.addConstrs(
        #     (
        #         dis[i] == 0.5 * s_star[i] * (div[i] - 1) * gp.nlfunc.log(div[i])
        #         for i in range(R)
        #     ),
        #     "c3",
        # )

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            log.critical(f"m.Status: {m.Status}")
            m.computeIIS()
            m.write("model1.ilp")
            raise RuntimeError("model is infeasible")

        flow_opt = np.array([[flow[i, j].x for j in range(R)] for i in range(R)])

        ## calculate policy
        qmat = np.divide(
            flow_opt,
            state[:, None],
            out=np.zeros_like(flow_opt),
            where=state[:, None] != 0,
        )
        np.fill_diagonal(qmat, 0)
        diag_value = 1 - np.sum(qmat, axis=1)
        np.fill_diagonal(qmat, diag_value)
        qmat = np.clip(qmat, 0, 1)
        policy = qmat / np.sum(qmat, axis=1, keepdims=True)
        # assert np.allclose(np.sum(policy, axis=1), 1)

        ## sample from policy
        routing = {}
        R = graph.n_nodes
        for i in range(R):
            if empty_cars[i] == 0:
                continue
            dests = dict(
                Counter(np.random.choice(np.arange(R), size=empty_cars[i], p=policy[i]))
            )
            routing.update({(i, k): v for k, v in dests.items() if k != i})

        log.debug(f"routing: {routing}")
        return routing

    def _state_lookahead(
        self,
        cur_state,
        future_flows,
        avg_demand_time,
        tt_f=None,
        tt_e=None,
        demand_mask=None,
        reb_mask=None,
    ) -> np.ndarray:
        # f_ahead = avg_demand_time
        # future_inflows = future_flows[:f_ahead]
        # if len(future_inflows) < f_ahead:
        #     future_inflows = np.pad(
        #         future_inflows,
        #         ((0, f_ahead - len(future_inflows)), (0, 0), (0, 0)),
        #         mode="constant",
        #     )
        # to_arrive = future_inflows.sum(axis=(0, 1, 2))

        future_flows = future_flows.sum(axis=(0))  # sum over time
        eps = 1e-6
        f_car_arrival = (
            future_flows[CarStatusEnum.FULL.value]
            # * demand_mask
            * np.reciprocal(tt_f + eps)
        )
        e_car_arrival = (
            future_flows[CarStatusEnum.EMPTY.value]
            # * reb_mask
            * np.reciprocal(tt_e + eps)
        )

        # sum over origin nodes
        to_arrive = np.sum(f_car_arrival + e_car_arrival, axis=0, dtype=int)
        return to_arrive

    def _demand_lookahead(self, state, historical_demands, d_ahead):
        d_ahead = self.d_ahead_steps
        historical_demands = historical_demands[-d_ahead:]
        if len(historical_demands) < d_ahead:
            length = len(historical_demands)
            historical_demands = np.pad(
                historical_demands,
                ((d_ahead - length, 0), (0, 0), (0, 0)),
                mode="edge",
            )

        lookahead_d = len(historical_demands)
        weights = np.power(self.demand_decay, np.flip(np.arange(lookahead_d)))
        w = historical_demands.sum(axis=2) * weights[:, None]
        target_state = w.sum(axis=0) / w.sum()
        log.warning(f"target: {target_state.sum()}, ori: {state.sum()}")
        assert np.isclose(target_state.sum(), state.sum())
        return target_state

    def _formulate_policy(self, flow_opt):
        pass


class ECRGRL(Algorithm):
    """reproduction of the GraphRL algorithm"""

    def __init__(
        self,
        mode: str = "test",
        dataset: str = None,
        ckpt_path: str | Path = None,
        use_cuda: bool = False,
        cplex_path: str = None,
        cplex_respath: str = None,
    ):
        self.algo_name = "ECR-GRL"
        self.mode = mode
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self._model = ECRA2C(input_size=21, device=device).to(device)
        self._ckptpath = ckpt_path
        self._cplex_path = cplex_path
        self._cplex_respath = cplex_respath
        self._dataset = dataset
        if self.mode == "train":
            self._model.train()
        elif self.mode in ["test", "equivalence_check"]:
            self._model.load_checkpoint(path=self._ckptpath)
            self._model.eval()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        log.info(f"ckpt path: {self._ckptpath}")

    def execute(
        self,
        obs: "ECRObs" = None,
    ):
        # get action from RL baseline
        if self.mode == "train":
            action_rl = self._model.select_action(obs)
        elif self.mode in ["test", "equivalence_check"]:
            concentration, _ = self._model(obs)
            action_rl = concentration / (concentration.sum() + 1e-16)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        action_rl = action_rl.detach().cpu().numpy()
        # log.warning(f"action_rl: {list(action_rl)}")

        # define current state and target state
        s = obs["available_cars"]
        R = len(s)
        M = s.sum()
        target_s = np.round(M * action_rl, decimals=1).astype(int)

        # formulate variables for LCP
        graph = obs["graph"]
        travel_time = graph.tt_e
        reb_mask = graph.reb_mask
        demand_mask = graph.demand_mask

        s_ = [(n, s[n]) for n in range(R)]
        target_s_ = [(n, target_s[n]) for n in range(R)]
        edgeAttr = [(i, j, travel_time[i, j]) for i, j in np.argwhere(reb_mask == 1)]

        # solve LCP
        action = ecr_lcp_solver(
            s=s_,
            target_s=target_s_,
            edges=np.argwhere(demand_mask == 1),
            edgeAttr=edgeAttr,
            t=obs["timestamp"],
            res_path=self._cplex_respath,
            CPLEXPATH=self._cplex_path,
        )

        return action

    def model_log_reward(self, reward):
        self._model.rewards.append(reward)

    def training_step(self, t):
        self._model.training_step(t=t)

    def ckpt_update(self):
        self._model.save_checkpoint(path=self._ckptpath)
