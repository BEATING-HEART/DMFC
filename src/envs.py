from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import chain, product
from typing import TYPE_CHECKING, Dict, Tuple, TypedDict

import gurobipy as gp
import numpy as np
import tomli
from gurobipy import GRB
from loguru import logger as log

from src.cplex.solver import ecr_matching_solver
from src.utils import fraction_allocation

if TYPE_CHECKING:
    from agents import SCIMAgentAction
    from graphs import ECRGraph, SCIMGraph


class BaseEnv:
    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        raise NotImplementedError

    def demand_step(self) -> None:
        raise NotImplementedError

    def action_step(self) -> None:
        raise NotImplementedError

    def _reposition_step(self) -> None:
        raise NotImplementedError

    def _conversion_step(self) -> None:
        raise NotImplementedError

    # def log(self) -> None:
    #     raise NotImplementedError

    # def _sample_demands(self) -> None:
    #     raise NotImplementedError


class SCIMObs(TypedDict):
    timestamp: int
    inflows: np.ndarray
    demands: np.ndarray
    node_holdings: np.ndarray
    pending_demands: np.ndarray
    future_inflows: np.ndarray  # future inflows of each node.
    historical_demands: np.ndarray  # history demands of each node.
    graph: "SCIMGraph"
    lcp_data: Dict[str, np.ndarray]


@dataclass
class SCIMEnvConfig:
    episode_length: int = 30
    is_waiting: bool = True
    time_redundancy: int = 5


class SCIMEnv(BaseEnv):
    def __init__(
        self, demand_pattern: str = "fixed", config: SCIMEnvConfig = SCIMEnvConfig()
    ) -> None:
        super().__init__()
        self._config = config
        self._t = 0
        self._demand = None
        self._demand_pattern = demand_pattern
        self._node_inflows = None
        """
        self._node_inflows:
            shape: (t, k, n). t: timestamp, k: commodities, n: nodes.
        """
        self._satisfied = None
        self._node_holdings = None
        """
        self._node_holdings: 
            shape: (k, n). k: commodities, n: nodes.
        """
        self._pending_demands = None
        self._info = {}

    @property
    def is_waiting(self):
        """
        This property indicates whether the customer will wait for the product.
        waiting: True - the customer will wait for the product.
        waiting: False - the customer will no wait for the product.
        """
        return self._config.is_waiting

    def reset(self, graph: "SCIMGraph"):
        self._t = 0

        nodes = graph.nodes
        lam_max = np.array([data.get("lambda_max", 0) for _, data in nodes])
        lam_var = np.array([data.get("lambda_var", 0) for _, data in nodes])

        L = self._config.episode_length + self._config.time_redundancy
        K = graph.n_commodities
        N = graph.n_nodes

        assert K == 1, "other cases not implemented"

        self._demand = self._demand_init(lam_max, lam_var, shape=(L, N))  # (L, K, N)
        self._node_inflows = np.zeros((L, N), dtype=int)  # (L, K, N)
        self._satisfied = np.zeros((L, N), dtype=int)  # (L, K, N)
        self._node_holdings = np.zeros((N), dtype=int)  # (K, N)
        self._pending_demands = np.zeros((N), dtype=int)  # (K, N)

    def demand_step(self, graph: "SCIMGraph"):
        """
        external demand satisfaction.
        """
        self._lcp_data = {
            "state": self._node_holdings - self._pending_demands,
            "lcp_state": self._node_holdings
            - self._pending_demands
            + self._node_inflows[self._t],
        }
        ## [1] arrival of inflows
        self._node_holdings += self._node_inflows[self._t]

        demand = self._demand[self._t]

        # only one commodity in our case.
        comm = graph.get_commodity_data(0)
        product_price = comm["product_price"]

        assert np.all(demand[graph.factory_nodes] == 0)

        self._pending_demands += demand
        self._satisfied[self._t] = np.minimum(
            self._node_holdings, self._pending_demands
        )
        self._node_holdings -= self._satisfied[self._t]
        self._pending_demands -= self._satisfied[self._t]

        if self._config.is_waiting:
            waiting_cnt = np.sum(self._pending_demands)
        else:
            self._pending_demands.fill(0)
            waiting_cnt = 0

        revenue = np.sum(demand * product_price)  # sold products

        obs: SCIMObs = self._obs(graph)
        self._info = {
            "revenue": revenue,
            "waiting_cnt": waiting_cnt,
            "waiting_panelty": waiting_cnt * product_price * 1.5,  # waiting panelty
            "demand_t": demand.squeeze(),
            "node_inflows": self._node_inflows[self._t].squeeze().copy(),
            "store_outflows": self._satisfied[self._t].squeeze().copy(),
        }
        return obs

    def action_step(self, graph: "SCIMGraph", action: "SCIMAgentAction"):
        prod, ship = action
        # log.debug(prod)
        # log.debug(ship)
        prod_cost = self._conversion_step(
            g=graph,
            prod=prod,
            demand=self._demand[self._t],
        )
        trans_cnt, trans_cost = self._reposition_step(graph=graph, ship=ship)

        # storage cost
        node_storage_cost = graph.get_node_attributes("storage_cost")
        node_storage_cost = np.array(list(node_storage_cost.values()))
        storage_cost = np.sum(self._node_holdings * node_storage_cost)

        # overstock panelty
        node_capacity = list(graph.get_node_attributes("storage_capacity").values())
        panelty = graph.get_commodity_data(0)["product_price"] * 1.5
        # log.info(self._node_holdings - np.array(node_capacity) > 0)  # no overstock
        overstock_cnt = np.sum(
            np.maximum(0, self._node_holdings - np.array(node_capacity))
        )
        overstock_panelty = overstock_cnt * panelty

        # throw away overstocked products.
        self._node_holdings = np.minimum(self._node_holdings, node_capacity)
        self._info.update(
            {
                "production": sum(prod.values()),
                "prod_cost": prod_cost,
                "factory_outflows": trans_cnt[graph.factory_nodes].copy(),
                "trans_cost": trans_cost,
                "overstock_cnt": overstock_cnt,
                "overstock_panelty": overstock_panelty,
                "node_capacity": node_capacity,
                "stock_level": self._node_holdings.copy(),
                "storage_cost": storage_cost,
            }
        )

        obs = self._obs(graph)
        # terminated = self._t >= self._config.episode_length   ## GRL paper terminate condition.
        terminated = self._t + 1 >= self._config.episode_length
        truncated = False

        self._t += 1
        if terminated or truncated:
            self._info.update({"demands": self._demand[: self._config.episode_length]})
            if self._config.is_waiting:
                self._info.update(
                    {"satisfied": self._satisfied[: self._config.episode_length]}
                )

        reward = (
            self._info["revenue"]
            - self._info["waiting_panelty"]
            - self._info["trans_cost"]
            - self._info["prod_cost"]
            - self._info["storage_cost"]
            - self._info["overstock_panelty"]
        )
        self._info.update({"reward": reward})

        outflows = np.array(
            list(
                chain(
                    self._info["factory_outflows"],
                    self._info["store_outflows"][graph.store_nodes],
                )
            )
        )
        self._info.update({"outflows": outflows})
        return obs, reward, terminated, truncated, self._info

    def _conversion_step(
        self,
        g: "SCIMGraph",
        prod: np.ndarray,
        demand: np.ndarray,
    ) -> float:
        prod_cost = 0

        ## get product. ()
        comm = g.get_commodity_data(0)
        production_time = comm["production_time"]
        production_cost = comm["production_cost"]

        ## production
        for n in g.factory_nodes:
            self._node_inflows[self._t + production_time, n] += prod[n]
            prod_cost += prod[n] * production_cost

        return prod_cost

    def _reposition_step(self, graph: SCIMGraph, ship: np.ndarray) -> float:
        edge_time = graph.get_edge_attributes("edge_time")
        edge_cost = graph.get_edge_attributes("edge_cost")
        edge_list = graph.edge_list

        trans_cost = 0
        trans_cnt = np.zeros(graph.n_nodes)
        for edge, flow in ship.items():  # edge, flow
            s, d = edge  # source, destination
            e_time = edge_time[edge]
            e_cost = edge_cost[edge]
            self._node_inflows[self._t + e_time, d] += flow
            self._node_holdings[s] -= flow
            trans_cost += e_cost * flow
            trans_cnt[s] += flow

        # for n in g.factory_nodes:
        #     holdings = self._node_holdings[0, n]
        #     if holdings == 0:
        #         continue

        #     out_edge_indices, prob = ship[n]
        #     # counter = Counter(np.random.choice(out_edge_indices, size=holdings, p=prob))
        #     # counter = fraction_allocation(holdings, out_edge_indices, prob)
        #     counter = fraction_allocation(
        #         holdings, len(out_edge_indices), prob, labels=out_edge_indices
        #     )
        #     outflows = np.array([counter[edge] for edge in out_edge_indices[:-1]])
        #     # outflows = np.round(holdings * prob)[:-1]  # no random
        #     # assert outflows.sum() == holdings
        #     trans_cnt[n] += outflows.sum()
        #     log.debug(f"outflows: {outflows}")

        #     for edge in out_edge_indices:
        #         if edge == -1:
        #             continue
        #         _, v = g.edge_list[edge]
        #         edge_data = g.get_edge_data(n, v)
        #         edge_time = edge_data["edge_time"]
        #         edge_cost = edge_data["edge_cost"]
        #         self._node_inflows[self._t + edge_time, 0, v] += outflows[edge]
        #         self._node_holdings[0, n] -= outflows[edge]
        #         trans_cost += edge_cost * outflows[edge]
        return trans_cnt, trans_cost

    def _demand_init(
        self, lam_max: np.ndarray, lam_var: np.ndarray, shape: Tuple[int, int, int]
    ):
        L, N = shape
        if self._demand_pattern == "fixed":
            demand = np.tile(lam_max // 2, (L, 1)).astype(int)  # [: target_shape[0]]
        elif self._demand_pattern == "gaussian":
            demand = lam_max // 2 + np.random.normal(
                0,
                lam_var,
                size=self._demand.shape,
            )
            demand = np.maximum(0, np.round(demand)).astype(int)
        elif self._demand_pattern == "cosine":
            T = self._config.episode_length
            coeff = np.zeros((L, N))
            for t, n in product(range(L), range(N)):
                if n == 0:  # n is factory node.
                    pass
                coeff[t, n] = 2 * n + t

            demand = np.round(
                lam_max // 2 + ((lam_max // 2) * np.cos(4 * np.pi * coeff / T))
            ).astype(int)
            bias = np.random.randint(0, lam_var + 1, size=(L, N))
            bias[:, 0] = 0
            demand += bias
            # log.warning(f"L: {L}, N: {N}, demand: {demand.sum()}")
        else:
            raise ValueError(f"Unknown demand pattern: {self._demand_pattern}")
        return demand

    def _obs(self, graph) -> SCIMObs:
        return SCIMObs(
            inflows=self._node_inflows,
            demands=self._demand,
            future_inflows=self._node_inflows[self._t + 1 :],
            historical_demands=self._demand[: self._t + 1],
            pending_demands=self._pending_demands,  # currently in waiting list
            node_holdings=self._node_holdings,
            timestamp=self._t,
            graph=graph,
            lcp_data=self._lcp_data,
        )


class ECRObs(TypedDict):
    timestamp: int = 0
    flows: np.ndarray = None
    demands: np.ndarray = None
    available_cars: np.ndarray = None
    future_flows: np.ndarray = None  # future inflows of each region.
    historical_demands: np.ndarray = None  # history demands of each region.
    historical_order_profits: np.ndarray = None  # history profits of each order.
    graph: "ECRGraph" = None


class CarStatusEnum(Enum):
    EMPTY = 0
    FULL = 1


@dataclass
class ECREnvConfig:
    episode_length: int = 200  # 200


class ECREnv(BaseEnv):
    def __init__(
        self,
        fuel: float,
        expt_name: str = "",
        cplexpath: str = "",
        cplex_respath: str = "",
        algo_mode: str = "",
        config: ECREnvConfig = ECREnvConfig(),
    ) -> None:
        super().__init__()
        self._config = config
        self._t = 0
        self._demand = None
        self._flows = None  # shape (T, 2, N, N)  2: empty cars, full cars
        # self._node_inflows = None  # shape: (T, 2, N)  2: empty cars, full cars
        self._satisfied = None
        self._available_cars = None
        self._info = {}
        self._fuel = fuel
        self._expt_name = expt_name
        self._cplexpath = cplexpath
        self._cplex_respath = cplex_respath
        self._algo_mode = algo_mode
        self.__random_state = None

    def reset(self, graph: "ECRGraph"):
        self._t = 0
        # nodes = graph.nodes
        T = graph.epi_len  # episode length
        L, R = graph.get_node_params(0)["plam"].shape  # L > T

        if self.__random_state is not None:
            np.random.set_state(self.__random_state)

        self._demand = np.zeros((L, R, R), dtype=int)
        self._demand = np.random.poisson(graph.plam).astype(int)

        self.__random_state = np.random.get_state()

        if self._expt_name == "shuffle":
            tmp = self._demand[100:]
            shifted = np.roll(tmp, shift=1, axis=1)
            self._demand[100:] = shifted

        log.info(f"demand dtype: {self._demand.dtype}")
        log.info(f"demand shape: {self._demand.shape}")
        log.info(f"demand sum: {self._demand.sum()}")
        self._demand *= graph.demand_mask  # connectivity matrix

        self._flows = np.zeros((L, 2, R, R), dtype=int)
        # self._node_inflows = np.zeros((L, 2, R), dtype=int)
        self._satisfied = np.zeros((T, R), dtype=int)
        self._available_cars = np.zeros(R, dtype=int)
        frac = graph.plam[0].sum(axis=1) / graph.plam[0].sum()
        self._available_cars = fraction_allocation(graph.n_cars, R, frac)

    def demand_step(self, graph: "ECRGraph"):
        self.process_inflow()

        demand = self._demand[self._t]
        price = graph.price[self._t]
        f_travel_time = np.round(1 / graph.mu_f).astype(int)

        d = np.argwhere(demand > 0)
        d = [(*row, demand[tuple(row)], price[tuple(row)]) for row in d]
        c = [(i, self._available_cars[i]) for i in range(graph.n_nodes)]
        f = ecr_matching_solver(
            t=self._t,
            edges=[tuple(row) for row in np.argwhere(graph.demand_mask > 0)],
            demandAttr=d,
            accTuple=c,
            CPLEXPATH=self._cplexpath,
            res_path=self._cplex_respath,
        )
        log.debug(f"matching: {f}")

        satisfied = np.zeros(graph.n_nodes, dtype=int)
        revenue = 0.0
        travel_time = 0.0
        for (i, j), v in f.items():
            assert int(v) == v
            v = int(v)
            self.process_outflow(i, j, v, f_travel_time[i, j], CarStatusEnum.FULL.value)
            satisfied[i] += v
            revenue += price[i, j] * v
            travel_time += f_travel_time[i, j] * v
        self._satisfied[self._t] = satisfied

        if np.sum(demand) > 0:
            step_fulfillment_rate = np.sum(satisfied) / np.sum(demand)
        else:
            step_fulfillment_rate = 0

        total_demand_t = self._demand[: self._t + 1].sum()
        total_satisfied_t = self._satisfied[: self._t + 1].sum()
        cum_fulfillment_rate = total_satisfied_t / total_demand_t

        obs: ECRObs = self._obs(graph)
        self._info = {
            "revenue": revenue,
            "full_car_travel_time": travel_time,
            "full_car_fuel_cost": travel_time * self._fuel * graph.tstep,
            "demands": demand,
            "satisfied": satisfied,
            "step_fulfillment_rate": step_fulfillment_rate,
            "cum_fulfillment_rate": cum_fulfillment_rate,
        }
        # log.info(f"paxreward: {revenue - travel_time * self._fuel * graph.tstep}")
        return obs

    def action_step(self, graph: "ECRGraph", action: Dict[Tuple[int, int], int]):
        # log.info(f"rebAction: {action}")
        travel_time = graph.tt_e
        repo_time = 0
        for (i, j), v in action.items():
            self.process_outflow(i, j, v, travel_time[i, j], CarStatusEnum.EMPTY.value)
            repo_time += travel_time[i, j] * v

        obs: ECRObs = self._obs(graph)
        self._info.update(
            {
                "empty_car_travel_time": repo_time,
                "empty_car_fuel_cost": repo_time * self._fuel * graph.tstep,
            }
        )

        reward = (
            self._info["revenue"]
            - self._info["empty_car_fuel_cost"]
            - self._info["full_car_fuel_cost"]
        )
        # log.info(f"rebreward: {-self._info['empty_car_fuel_cost']}")
        self._info.update({"reward": reward})

        terminated = self._t + 1 >= self._config.episode_length
        truncated = False
        self._t += 1

        return obs, reward, terminated, truncated, self._info

    def process_inflow(self):
        flow = self._flows[self._t]
        node_inflows = flow.sum(axis=(0, 1))  # summation of type and origin region.
        self._available_cars += node_inflows

        # log.debug(f"inflows: {self._node_inflows[self._t].sum()}")
        # self._available_cars += self._node_inflows[self._t].sum(axis=0)

    def process_outflow(
        self, origin: int, dest: int, flow: int, latency: float, type: int
    ):
        """outflow departure.

        Args:
            origin (int): origin node.
            dest (int): destination node.
            flow (int): amount of resource to be transferred.
            latency (float): travel time.
            type (int): car status.
        """
        arrival_time = self._t + latency
        schedule_time = int(arrival_time) + 1
        # the car arrives in the middle of a time slot and could
        # be scheduled in the beginning of the next time slot.

        if schedule_time >= self._demand.shape[0]:  # out of time horizon
            return
        self._flows[schedule_time, type, origin, dest] += flow
        # self._node_inflows[schedule_time, type, dest] += flow
        self._available_cars[origin] -= flow

    def _obs(self, graph) -> ECRObs:
        return ECRObs(
            flows=self._flows,
            demands=self._demand,
            future_flows=self._flows[self._t + 1 :],
            historical_demands=self._demand[: self._t + 1],
            available_cars=self._available_cars,
            historical_order_profits=None,
            timestamp=self._t,
            graph=graph,
        )


if __name__ == "__main__":
    import json
    import os

    from graphs import SCIMGraph

    path = os.path.join(os.path.dirname(__file__), "scim.json")
    with open(path, "r") as f:
        data = json.load(f)

    data = data["1f2s"]

    g = SCIMGraph(config=None, data=data)
    config = SCIMEnvConfig(episode_length=10, comm_initial_amount=0)
    env = SCIMEnv(config=config, graph=g)
    env.reset(graph=g)
