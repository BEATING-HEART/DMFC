from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, TypedDict

import numpy as np

from src.envs import ECRObs, SCIMObs

if TYPE_CHECKING:
    from src.algorithms import Algorithm


class BaseAgent:
    def __init__(self) -> None:
        pass

    def sample_action(self) -> None:
        raise NotImplementedError


class SCIMAgentAction(TypedDict):
    factory_nodes: List[int]
    prod: Dict
    ship: Dict
    """
    shipment
        key: factory id (- commodity id).
        value: (out edge ids, outflow probability of each out edge)
    """


@dataclass
class SCIMAgentConfig:
    inflow_ahead: int = 3
    demand_ahead: int = 10
    demand_decay: float = 0.95


class SCIMAgent(BaseAgent):
    def __init__(
        self, algo: "Algorithm", config: SCIMAgentConfig = SCIMAgentConfig()
    ) -> None:
        super().__init__()
        self.algo = algo
        self._config = config

    def sample_action(self, obs: "SCIMObs") -> SCIMAgentAction:
        """sample action from the agent.

        Args:
            obs (SCIMObs): environment observation
            algo (str): algorithm name
        """

        ## preprocess
        inflow_steps = self._config.inflow_ahead
        demand_steps = self._config.demand_ahead

        future_inflows = obs["future_inflows"][:inflow_steps]
        historical_demands = obs["historical_demands"][-demand_steps:]
        _, n_nodes = future_inflows.shape

        # completion
        if len(future_inflows) < inflow_steps:
            future_inflows = np.pad(
                future_inflows,
                ((0, inflow_steps - len(future_inflows)), (0, 0)),
                mode="constant",
            )
        if len(historical_demands) < demand_steps:
            historical_demands = np.pad(
                historical_demands,
                ((demand_steps - len(historical_demands), 0), (0, 0)),
                mode="edge",
            )

        graph = obs["graph"]
        inventory_capacity = graph.get_node_attributes("storage_capacity").values()
        inventory_capacity = np.array(list(inventory_capacity))

        ## sample action
        prod, ship = self.algo.execute(
            graph=graph,
            current_inventory=obs["node_holdings"],
            inventory_capacity=inventory_capacity,
            historical_demands=historical_demands,
            pending_demand=obs["pending_demands"],
            resource_inflow=future_inflows,
            obs=obs,
        )
        action = (prod, ship)  # action

        return action


class ECRAgentAction(TypedDict):
    repositioning_policy: np.ndarray


@dataclass
class ECRAgentConfig:
    demand_ahead: int = 10
    inflow_ahead: int = None  # set when dataset is loaded
    demand_decay: float = 0.95
    eps: float = 1e-6
    """small value for numerical stability"""
    # use_cuda: bool = False


class ECRAgent(BaseAgent):
    def __init__(
        self, algo: "Algorithm", config: ECRAgentConfig = ECRAgentConfig()
    ) -> None:
        super().__init__()
        self.algo = algo
        self._config = config

    def sample_action(self, obs: "ECRObs") -> None:
        action = self.algo.execute(obs=obs)
        return action
