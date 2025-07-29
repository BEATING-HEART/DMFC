from collections import defaultdict
from copy import deepcopy
from itertools import product

import networkx as nx
import numpy as np
from loguru import logger as log


class Network:
    def __init__(
        self,
        graph_path=None,
        G=None,
        dmax=[10, 10, 10],
        dvar=[2, 2, 2],
        tf=20,
        sd=None,
        randomize_graph_args=(None, None),
        randomize_demand_args=(None, None),
        factory_nodes=[0],
        warehouse_nodes=[1],
        product_prices=[15],
        production_costs=[5],
        storage_capacities=[5, 10],
        storage_costs=[2, 1],
        edge_costs=[1],
        edge_time=[1],
        production_time=1,
        penalty_cost=None,
    ):
        # book-keeping variables
        self.sd = sd
        self.factory = factory_nodes
        self.warehouse = warehouse_nodes
        self.product_prices = product_prices
        self.production_costs = production_costs
        self.production_time = production_time
        self.storage_capacities = storage_capacities
        self.storage_costs = storage_costs
        self.edge_costs = edge_costs
        self.edge_time = edge_time
        self.penalty_cost = (
            1.5 * self.product_prices[0] if penalty_cost is None else penalty_cost
        )
        self.randomize_graph_args = randomize_graph_args
        self.randomize_demand_args = randomize_demand_args
        self.tf = tf
        self.dmax = dmax
        self.dvar = dvar
        self.demand = defaultdict(lambda: defaultdict(int))
        # if availabke, load graph
        self.load_graph(graph_path, G)

        # self.randomize_graph(self.randomize_graph_args[0], self.randomize_graph_args[1])
        self.get_random_demand(
            self.randomize_demand_args[0], self.randomize_demand_args[1]
        )
        # set node and edge properties
        self._set_node_properties()
        self._set_edge_properties()

    def load_graph(self, graph_path=None, G=None):
        """
        Load a pre-defined graph.
        A graph is defines nodes and edges (with travel times and cost associated)
        """
        self.G = nx.read_graphml(graph_path) if graph_path is not None else G

    def randomize_graph(self, randomize_graph_fn=None, tt_logic=None):
        self.random_graph = deepcopy(self.G)
        if randomize_graph_fn is None:
            if tt_logic == "random-tt":
                for e_i, e in enumerate(self.G.edges):
                    if e not in self.random_graph.edges:
                        self.random_graph.add_edge(e)
                    # self.random_graph.edges[e]['time'] = max(1, np.random.randint(1,7))
                    self.random_graph.edges[e]["edge_time"] = self.edge_time[e_i]
                    self.random_graph.edges[e]["edge_cost"] = self.edge_costs[e_i]
        else:
            self.random_graph = randomize_graph_fn(self)
        return self.random_graph

    def get_random_graph(self):
        # output is self.G
        return NotImplemented

    # TODO
    # 1. Select OD(s) + Absolute demand
    def get_random_demand(self, randomize_demand_fn=None, demand_logic=None):
        # get curves for all warehouse nodes
        if randomize_demand_fn is None:
            for t in range(self.tf):
                for node in self.G:
                    self.demand[t][node] = 0
                for warehouse in self.warehouse:
                    demand = self.get_demand_curve(
                        self.factory[0], warehouse
                    )  # TODO: fix for different factories/products
                    self.demand[t][warehouse] = demand[t]
        else:
            self.demand = randomize_demand_fn(self)
        return self.demand

    def get_demand_curve(self, j_prod, i_warehouse):
        j = j_prod + 1
        demand = []
        for t in range(self.tf):
            demand.append(
                np.round(
                    self.dmax[i_warehouse - 1] / 2
                    + self.dmax[i_warehouse - 1]
                    / 2
                    * np.cos(4 * np.pi * (2 * j * i_warehouse + t) / self.tf)
                    + np.random.randint(0, self.dvar[i_warehouse - 1] + 1)
                )
            )
        return demand

    def _set_node_properties(self):
        for i, node in enumerate(self.G.nodes):
            self.G.nodes[node]["storage_capacity"] = self.storage_capacities[i]
            self.G.nodes[node]["storage_cost"] = self.storage_costs[i]
            if node in self.factory:
                self.G.nodes[node]["production_cost"] = self.production_costs[node]

    def _set_edge_properties(self):
        pass


class SupplyChainIventoryManagement:
    # initialization
    def __init__(self, scenario):
        self.scenario = deepcopy(scenario)  # copy of network object
        self.G = scenario.G  # newtorkx object representing the current scenario
        self.time = 0  # current time
        self.tf = scenario.tf  # episode lenght
        self.demand = self.scenario.demand  # demand curves
        self.nodes = list(self.G)  # list of all nodes
        self.acc = defaultdict(
            lambda: defaultdict(int)
        )  # current accumulation in every node
        self.dacc = defaultdict(lambda: defaultdict(int))  # future accumulation
        self.flow = defaultdict(
            lambda: defaultdict(int)
        )  # the flow currenty active along all edges
        self.prod = defaultdict(
            lambda: defaultdict(int)
        )  # the production level at factories
        self.arrival_flow = defaultdict(
            lambda: defaultdict(int)
        )  # the flow meant to arrive
        self.arrival_prod = defaultdict(
            lambda: defaultdict(int)
        )  # the production meant to finish
        self.num_nodes = len(scenario.G)  # number of nodes
        t = self.time  # current timestep

        # add the initialization of info here
        self.info = dict.fromkeys([""], 0)
        self.reward = 0
        self.done = False

    def step(self, action):
        t = self.time
        # Unpack the action: a = (production, shipping)
        prod_a, ship_a = action[0], action[1]
        self.flow[t] = ship_a
        self.prod[t] = prod_a

        # compute stocks before production & flow
        for n in self.scenario.G.nodes:
            # calculate factory stocks as: min(stocks_t + production_{t-prod_time} - shipped_t, storage_capacity)
            if n in self.scenario.factory:
                self.arrival_prod[t + self.scenario.production_time][n] += self.prod[t][
                    n
                ]
                self.acc[t][n] = (
                    self.acc[t - 1][n] + self.prod[t - self.scenario.production_time][n]
                )
            # calculate warehouse stocks as: min(stock_t + shipped_{t-tt} - demand_t, storage_capacity)
            if n in self.scenario.warehouse:
                # print("n: ", n, self.arrival_flow[t][n], self.demand[t][n])
                self.acc[t][n] = self.acc[t - 1][n] + self.arrival_flow[t][n]

        # transform flow into arrival_flow and compute total departing flow for each node
        depart_flow = defaultdict(float)
        for e in self.flow[t]:
            self.flow[t][e] = min(self.flow[t][e], self.acc[t][e[0]])
            tt = self.random_graph.edges[e]["edge_time"]
            self.arrival_flow[t + tt][e[1]] += self.flow[t][e]
            assert self.flow[t][e] <= self.random_graph.edges[e]["capacity"]
            depart_flow[e[0]] += self.flow[t][e]

        for n in self.scenario.G.nodes:
            # calculate factory stocks as: min(stocks_t + production_{t-prod_time} - shipped_t, storage_capacity)
            if n in self.scenario.factory:
                self.acc[t][n] = (
                    self.acc[t - 1][n]
                    + self.prod[t - self.scenario.production_time][n]
                    - depart_flow[n]
                )
            # calculate warehouse stocks as: min(stock_t + shipped_{t-tt} - demand_t, storage_capacity)
            if n in self.scenario.warehouse:
                # print("n: ", n, self.arrival_flow[t][n], self.demand[t][n])
                self.acc[t][n] = (
                    self.acc[t - 1][n] + self.arrival_flow[t][n] - self.demand[t][n]
                )

        self.obs = (self.random_graph, self.arrival_flow, self.demand)

        # compute cost (1) transportation cost: flow_ij*cost_ij, (2) storage cost: stock*cost
        # (3) production cost: prod*cost, (4) penalty cost for understock: negative stock*penalty
        cost = 0
        transport_cost = 0
        storage_cost = 0
        production_cost = 0
        penalty_outstock_cost = 0
        penalty_storage_cost = 0
        for e in self.flow[t]:
            transport_cost += self.flow[t][e] * self.random_graph.edges[e]["edge_cost"]
        for n in self.scenario.G.nodes:
            storage_cost += (
                max(self.acc[t][n], 0) * self.scenario.G.nodes[n]["storage_cost"]
            )
            if n in self.scenario.factory:
                production_cost += (
                    self.prod[t][n] * self.scenario.G.nodes[n]["production_cost"]
                )
            if self.acc[t][n] < 0:
                penalty_outstock_cost -= self.acc[t][n] * self.scenario.penalty_cost
            if self.acc[t][n] > self.scenario.G.nodes[n]["storage_capacity"]:
                penalty_storage_cost += (
                    self.acc[t][n] - self.scenario.G.nodes[n]["storage_capacity"]
                ) * self.scenario.penalty_cost

        cost = (
            transport_cost
            + storage_cost
            + production_cost
            + penalty_outstock_cost
            + penalty_storage_cost
        )
        self.info["transport_cost"] = transport_cost
        self.info["storage_cost"] = storage_cost
        self.info["production_cost"] = production_cost
        self.info["penalty_outstock_cost"] = penalty_outstock_cost
        self.info["penalty_storage_cost"] = penalty_storage_cost

        # compute revenue (product sold * price)
        revenue = 0
        for n in self.scenario.warehouse:
            revenue += self.demand[t][n] * self.scenario.product_prices[0]
        self.reward = revenue - cost

        # compute stocks after production & flow
        for n in self.scenario.G.nodes:
            # calculate factory stocks as: min(stocks_t + production_{t-prod_time} - shipped_t, storage_capacity)
            if n in self.scenario.factory:
                self.acc[t][n] = min(
                    self.acc[t][n], self.scenario.G.nodes[n]["storage_capacity"]
                )
            # calculate warehouse stocks as: min(stock_t + shipped_{t-tt} - demand_t, storage_capacity)
            if n in self.scenario.warehouse:
                # print("n: ", n, self.arrival_flow[t][n], self.demand[t][n])
                self.acc[t][n] = min(
                    self.acc[t][n], self.scenario.G.nodes[n]["storage_capacity"]
                )

        # 4. Termination criterion
        if t + 1 >= self.tf:
            self.done = True

        self.time += 1
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self.random_graph = self.scenario.randomize_graph(
            self.scenario.randomize_graph_args[0], self.scenario.randomize_graph_args[1]
        )
        self.demand = self.scenario.get_random_demand(
            self.scenario.randomize_demand_args[0],
            self.scenario.randomize_demand_args[1],
        )
        self.acc = defaultdict(
            lambda: defaultdict(int)
        )  # current accumulation in every node
        self.dacc = defaultdict(lambda: defaultdict(int))  # future accumulation
        self.flow = defaultdict(
            lambda: defaultdict(int)
        )  # the flow currenty active along all edges
        self.prod = defaultdict(
            lambda: defaultdict(int)
        )  # the production level at factories
        self.arrival_flow = defaultdict(lambda: defaultdict(int))
        self.arrival_prod = defaultdict(lambda: defaultdict(int))
        self.obs = (self.random_graph, self.arrival_flow, self.demand)
        self.time = 0
        self.done = False
        return self.obs


class MySCIM(SupplyChainIventoryManagement):
    def __init__(self, scenario):
        super().__init__(scenario)

    @property
    def edge_list(self):
        return list(self.G.edges())


class MyNetwork(Network):
    def __init__(self, data):
        # log.info(data.keys())
        # edge_list = [(d["source"], d["dest"]) for d in data["edges"]]
        G = nx.DiGraph()

        nodes = data["nodes"]
        nodes = list(map(lambda n: tuple(n.values()), nodes))
        G.add_nodes_from(nodes)

        edges = data["edges"]
        edges = list(map(lambda e: tuple(e.values()), edges))
        G.add_edges_from(edges)
        for e in G.edges:
            G.edges[e]["capacity"] = 1000

        node_types = nx.get_node_attributes(G, "type")
        factory_nodes = [k for k, v in node_types.items() if v == "factory"]
        warehouse_nodes = [k for k, v in node_types.items() if v == "store"]

        dmax = [nx.get_node_attributes(G, "lambda_max")[i] for i in warehouse_nodes]
        dvar = [nx.get_node_attributes(G, "lambda_var")[i] for i in warehouse_nodes]
        storage_costs = [nx.get_node_attributes(G, "storage_cost")[i] for i in G.nodes]
        storage_capacities = [
            nx.get_node_attributes(G, "storage_capacity")[i] for i in G.nodes
        ]
        edge_costs = [G.get_edge_data(*e)["edge_cost"] for e in G.edges]
        edge_time = [G.get_edge_data(*e)["edge_time"] for e in G.edges]

        commodities = data["commodities"]
        commodity = commodities[0]
        production_time = commodity["values"]["production_time"]
        product_prices = [commodity["values"]["product_price"]]
        production_costs = [commodity["values"]["production_cost"]]

        tf = 30  # episode length

        self.G = G
        self.factory = factory_nodes
        self.warehouse = warehouse_nodes
        self.product_prices = product_prices
        self.production_costs = production_costs
        self.production_time = production_time
        self.storage_capacities = storage_capacities
        self.storage_costs = storage_costs
        self.edge_costs = edge_costs
        self.edge_time = edge_time
        self.penalty_cost = 1.5 * self.product_prices[0]
        self.randomize_graph_args = (None, "random-tt")
        self.randomize_demand_args = (None, "single-od")
        self.tf = tf
        self.dmax = dmax
        self.dvar = dvar
        self.demand = defaultdict(lambda: defaultdict(int))

        self._set_node_properties()
        self._set_edge_properties()

    def get_random_demand(self, randomize_demand_fn=None, demand_logic=None):
        T = self.tf
        eps = 5
        L = T + eps
        K = 1
        N = len(self.G)
        # logger.debug(np.random.rand(5))
        coeff = np.zeros((L, N))
        for t, n in product(range(L), range(N)):
            if n == 0:  # n is factory node.
                pass
            coeff[t, n] = 2 * n + t

        lam_var = np.array([0] + self.dvar)
        lam_max = np.array([0] + self.dmax)

        # log.debug(bias)
        demand = np.round(
            lam_max // 2 + (lam_max // 2) * np.cos(4 * np.pi * coeff / T)
        ).astype(int)
        bias = np.random.randint(0, lam_var + 1, size=(L, N))
        bias[:, 0] = 0
        demand += bias
        # log.warning(f"L: {L}, N: {N}, demand: {demand.sum()}")

        d = defaultdict(lambda: defaultdict(int))
        for t in range(L):
            for warehouse in self.warehouse:
                d[t][warehouse] = demand[t, warehouse]

        self.demand = d
        return d
