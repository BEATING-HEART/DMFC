from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tomli


def load_config(config_path: Path, key: str) -> str:
    p = config_path.suffix
    if p == ".toml":
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        return config[key]
    else:
        raise ValueError("unsupported config file format")


def load_ecr_data(
    dataset_name: str,
    T: int = 200,
    start_hour: int = 13,
    delta: int = 15,
    beta: float = None,
    is_unit_price: bool = False,
) -> Dict:
    ecr_path = Path(__file__).parent.parent / "data" / "ecr" / dataset_name
    mu_e = np.load(ecr_path / "mu_e.npy")
    mu_f = np.load(ecr_path / "mu_f.npy")
    plam = np.load(ecr_path / "plam.npy")
    price = np.load(ecr_path / "prices.npy")
    adjacency = np.load(ecr_path / "adjacency.npy")

    with open(ecr_path / "data.json", "r") as f:
        data = json.load(f)
    C = data["car_num"]
    tstep = data["tstep"]

    # if beta is not None:
    #     fuel = beta
    # else:
    #     fuel = data["fuel_coeff"]

    if is_unit_price:
        price = np.ones_like(price).astype(int)
        # fuel = 0.0

    # fuel = fuel * tstep  # fuel cost coeff

    # full car fuel cost
    # fuel_cost = fuel * np.round(1 / mu_f)
    # order_profit = price - fuel_cost
    # assert np.all(order_profit >= 0), "order profit should be positive"

    start = start_hour * 6  # start slow time
    L = T + delta  # episode length (T) + redundant data (delta)
    R = adjacency.shape[0]
    repeat = (L + start) // plam.shape[0] + 1
    plam = np.tile(plam, (repeat, 1, 1))
    price = np.tile(price, (repeat, 1, 1))
    plam = plam[start : start + L]
    price = price[start : start + L]

    data = {
        "dataset": dataset_name,
        "mu_e": mu_e,
        "mu_f": mu_f,
        "plam": plam,
        "price": price,
        "adjacency": adjacency,
        "R": R,
        "T": T,  # episode length
        "C": C,
        "tstep": tstep,
        # "fuel": fuel,  # fuel cost coeff
        # "order_profit": order_profit,
    }
    return data


def load_scim_data(dataset_name: str) -> Dict:
    path = Path(__file__).parent.parent / "data" / "scim" / "scim.json"
    # path = os.path.join(os.path.dirname(__file__), "data", "scim", "scim.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data[dataset_name]


def fraction_allocation(
    n: int, m: int, p: List[float], labels: List[int] = None
) -> List[int] | Dict[int, int]:
    # labels: List[int],
    """
    allocate n items to m groups according to the fraction p
    return the number of items allocated to each group
    """
    assert len(p) == m
    assert np.isclose(sum(p), 1.0)

    cdf = np.cumsum(p)
    values = np.linspace(0, 1, n, endpoint=False) + 1 / (2 * n)
    indices = np.searchsorted(cdf, values, side="right")
    allocations = np.bincount(indices, minlength=m)
    if labels is not None:
        return {labels[i]: allocations[i] for i in range(m)}
    return allocations


def random_rounding(x: np.ndarray | List[float] | float) -> np.ndarray:
    x = np.asarray(x)
    fractional, integer = np.modf(x)
    rand = np.random.rand(*x.shape) < fractional
    return int(integer + rand)
