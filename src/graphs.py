from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict

import networkx as nx
import numpy as np


@dataclass
class GraphConfig:
    pass


class BaseGraph:
    def __init__(self) -> None:
        # self._config = config
        self._graph: nx.DiGraph = None

    @property
    def nodes(self):
        """return networkx NodeView"""
        return self._graph.nodes(data=True)

    @property
    def edges(self):
        """return networkx EdgeView"""
        return self._graph.edges(data=True)

    @property
    def selfloop_edges(self):
        return nx.selfloop_edges(self._graph, data=True)

    @property
    def node_list(self) -> List[int]:
        """list of nodes"""
        return list(self._graph.nodes())

    @property
    def edge_list(self) -> List[Tuple[int, int]]:
        """list of edges"""
        return list(self._graph.edges())

    @property
    def n_nodes(self) -> int:
        """num of nodes"""
        return self._graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """num of edges"""
        return self._graph.number_of_edges()

    @property
    def adj_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self._graph).astype(int)

    def in_edges(self, node: int) -> List[Tuple[int, int]]:
        """list of in_edges"""
        return list(self._graph.in_edges(node))

    def out_edges(self, node: int) -> List[Tuple[int, int]]:
        """list of out_edges"""
        return list(self._graph.out_edges(node))

    def in_edges_indices(self, node: int) -> List[int]:
        """list of in_edge indices"""
        edges = self.in_edges(node)
        return [self.edge_list.index(e) for e in edges]

    def out_edges_indices(self, node: int) -> List[int]:
        """list of out_edge indices"""
        edges = self.out_edges(node)
        return [self.edge_list.index(e) for e in edges]

    def get_node_data(self, node: int) -> Dict:
        return self._graph.nodes[node]

    # def get_edge_data(self, u: int, v: int):
    #     return self._graph.get_edge_data(u, v)

    def get_edge_data(self, edge: Tuple[int, int]):
        """get data of an edge."""
        return self._graph.get_edge_data(edge)

    def get_node_attributes(self, attribute: str) -> Dict:
        """
        args:
            attribute: Attribute name

        return:
            Dictionary of attributes keyed by node.
        """
        attrs = nx.get_node_attributes(self._graph, attribute)
        return attrs

    def get_edge_attributes(self, attribute: str) -> Dict:
        """
        args:
            attribute: Attribute name

        return:
            Dictionary of attributes keyed by edge.
        """
        attrs = nx.get_edge_attributes(self._graph, attribute)
        return attrs

    def get_edges_data_v2(
        self,
        edges: Tuple[int, int] | List[Tuple[int, int]],
        attribute: str,
    ):
        pass


@dataclass
class SCIMGraphConfig:
    r""" """


class SCIMGraph(BaseGraph):
    # class SCIMGraph(BaseGraph):
    def __init__(self, data: Dict, config: SCIMGraphConfig = None) -> None:
        super().__init__()
        self._graph = nx.DiGraph()

        nodes = data["nodes"]
        nodes = list(map(lambda n: tuple(n.values()), nodes))
        self._graph.add_nodes_from(nodes)

        edges = data["edges"]
        edges = list(map(lambda e: tuple(e.values()), edges))
        self._graph.add_edges_from(edges)

        commodities = data["commodities"]
        self._commodities = list(map(lambda c: tuple(c.values()), commodities))

    @property
    def commodities(self) -> List:
        return self._commodities

    @property
    def n_commodities(self) -> int:
        return len(self._commodities)

    @property
    def factory_nodes(self) -> List[int]:
        return [node for node, attr in self.nodes if attr.get("type") == "factory"]

    @property
    def n_factories(self) -> int:
        return len(self.factory_nodes)

    @property
    def store_nodes(self) -> List[int]:
        return [node for node, attr in self.nodes if attr.get("type") == "store"]

    @property
    def n_stores(self) -> int:
        return len(self.store_nodes)

    def get_commodity_data(self, idx: int) -> Dict:
        assert idx < self.n_commodities, "Commodity index out of range"
        return self._commodities[idx][1]

    @property
    def edge_list_bidirectional(self) -> List[Tuple[int, int]]:
        lst = list(self._graph.edges())
        lst_bi = lst + [(v, u) for u, v in lst]
        return lst_bi

    # def out_edges(self, node: int) -> List[Tuple[int, int]]:
    #     return list(self._graph.out_edges(node))

    # def in_edges(self, node: int) -> List[Tuple[int, int]]:
    #     return list(self._graph.in_edges(node))

    # def out_edges_indices(self, node: int) -> List[int]:
    #     return [self.edge_list.index(edge) for edge in self.out_edges(node)]

    # def in_edges_indices(self, node: int) -> List[int]:
    #     return [self.edge_list.index(edge) for edge in self.in_edges(node)]

    # @property
    # def is_sparse(self) -> bool:
    #     return self.n_edges < self.n_nodes * np.log(self.n_nodes)


class ECRNodeParams(TypedDict):
    node: int
    neighbors: np.ndarray
    mu_e: np.ndarray
    mu_f: np.ndarray
    plam: np.ndarray
    price: np.ndarray


class ECRGraph(BaseGraph):
    def __init__(self, data: Dict) -> None:
        super().__init__()
        # super().copy_docstrings()

        self._graph = nx.DiGraph()
        self._data = data

        self._graph.add_nodes_from(range(self._data["R"]))
        # print(self._graph.nodes(data=True))

        np.fill_diagonal(self._data["adjacency"], 1)
        for i, j in np.argwhere(self._data["adjacency"] == 1):
            self._graph.add_edge(
                i,
                j,
                mu_e=self._data["mu_e"][i, j],
                mu_f=self._data["mu_f"][i, j],
                plam=self._data["plam"][:, i, j],
                price=self._data["price"][:, i, j],
            )
        np.fill_diagonal(self._data["adjacency"], 0)

    @property
    def dataset(self) -> str:
        return self._data["dataset"]

    @property
    def epi_len(self) -> int:
        return self._data["T"]

    @property
    def n_cars(self) -> int:
        return self._data["C"]

    @property
    def plam(self) -> np.ndarray:
        return self._data["plam"]

    @property
    def price(self) -> np.ndarray:
        return self._data["price"]

    @property
    def mu_e(self) -> np.ndarray:
        return self._data["mu_e"]

    @property
    def mu_f(self) -> np.ndarray:
        return self._data["mu_f"]

    @property
    def tt_e(self) -> np.ndarray:
        return np.maximum(1, np.round(1 / self._data["mu_e"]).astype(int))

    @property
    def tt_f(self) -> np.ndarray:
        return np.maximum(1, np.round(1 / self._data["mu_f"]).astype(int))

    @property
    def avg_demand_time(self) -> int:
        mean_plam = self.plam.mean(axis=0)
        demand_time = np.round(1 / self._data["mu_f"])
        assert np.count_nonzero(mean_plam[demand_time == 999]) == 0
        return int(np.sum(mean_plam * demand_time) / np.sum(mean_plam))

    @property
    def avg_rebalance_time(self) -> int:
        pass

    @property
    def adj_matrix(self) -> np.ndarray:
        adj = nx.to_numpy_array(self._graph).astype(int)
        np.fill_diagonal(adj, 0)
        assert np.equal(adj, self._data["adjacency"]).all()
        return adj

    @property
    def demand_mask(self) -> np.ndarray:
        """demand mask support self loop"""
        return nx.to_numpy_array(self._graph).astype(int)

    @property
    def reb_mask(self) -> np.ndarray:
        adj = nx.to_numpy_array(self._graph).astype(int)
        np.fill_diagonal(adj, 0)
        assert np.equal(adj, self._data["adjacency"]).all()
        return adj

    # @property
    # def profit(self) -> np.ndarray:
    #     return self._data["order_profit"]

    @property
    def tstep(self) -> float:
        return self._data["tstep"]

    def get_node_params(self, node: int):
        """get parameters of all edges from node"""
        neighbors = np.array(list(self._graph.neighbors(node)))

        p_lam = self._data["plam"][:, node, :]
        mu_e = self._data["mu_e"][node]
        mu_f = self._data["mu_f"][node]
        price = self._data["price"][:, node, :]

        return ECRNodeParams(
            node=node,
            neighbors=neighbors,
            mu_e=mu_e,
            mu_f=mu_f,
            plam=p_lam,
            price=price,
        )


if __name__ == "__main__":
    import json
    from itertools import chain
    from pathlib import Path

    # g = nx.DiGraph()

    path = Path(__file__).parent.parent / "data" / "scim" / "scim.json"
    with open(path, "r") as f:
        data = json.load(f)

    data = data["1f10s"]
    g = SCIMGraph(data=data)
    # print(g.nodes)
    # print(g.node_list)
    # print(type(g.node_list[0]))
    # print(g.edges)
    # print(g.edge_list)
    # print(type(g.edge_list[0][0]))
    # print(g.adj_matrix)
    # print(g.get_in_edges(1))
    # print(g.get_out_edges(0))
    # # print(g.get_edge_data((0, 1)))
    # print(g.get_edge_attributes("edge_time"))
    # print(g.get_node_attributes("type"))
    # print(g.get_commodity_data(1))

    # nodes = data["nodes"]
    # nodes = list(map(lambda n: tuple(n.values()), nodes))
    # g.add_nodes_from(nodes)

    # edges = data["edges"]
    # edges = list(map(lambda e: tuple(e.values()), edges))
    # g.add_edges_from(edges)

    # commodities = data["commodities"]
    # commodities = list(map(lambda c: tuple(c.values()), commodities))
    # print(commodities)
    # print(g.nodes(data=True))

    # with open("./src/scim_bak.json", "r") as f:
    #     data = json.load(f)

    # data = data["1f2s"]
    # data = list(chain(*data.values()))
    # g.add_nodes_from(data)
    # print(g.nodes(data=True))

    # ecr_path = Path(__file__).parent.parent / "data" / "ecr" / "didi20"
    # mu_e = np.load(ecr_path / "mu_e.npy")
    # mu_f = np.load(ecr_path / "mu_f.npy")
    # plam = np.load(ecr_path / "plam.npy")
    # price = np.load(ecr_path / "prices.npy")
    # adjacency = np.load(ecr_path / "adjacency.npy")

    # T = 200
    # delta = 15
    # start = 6 * 13
    # L = T + delta
    # repeat = (L + start) // plam.shape[0] + 1
    # plam = np.tile(plam, (repeat, 1, 1))
    # price = np.tile(price, (repeat, 1, 1))
    # plam = plam[start : start + L]
    # price = price[start : start + L]

    # print(mu_e.shape, mu_f.shape, plam.shape, price.shape, adjacency.shape)

    # data = {
    #     "mu_e": mu_e,
    #     "mu_f": mu_f,
    #     "plam": plam,
    #     "price": price,
    #     "adjacency": adjacency,
    #     "T": T,
    # }

    # graph = ECRGraph(config=None, data=data)
    # print(graph.get_params(0))
