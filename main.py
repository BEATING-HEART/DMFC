from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tomli
import torch

# import tomli
from src.agents import ECRAgent, SCIMAgent
from src.algorithms import ECRDMFC, ECRGRL, SCIMDMFC, SCIMGRL
from src.envs import ECREnv, SCIMEnv
from src.graphs import ECRGraph, SCIMGraph
from src.loggers import ECRLogger, SCIMLogger
from src.simulator import Simulator
from src.utils import load_ecr_data, load_scim_data


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="scim",
        choices=["scim", "ecr"],
        help="set scenario",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1f2s",
        help="set dataset",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="fluid",
        help="define the algo to use",
    )
    parser.add_argument(
        "--algo_mode",
        type=str,
        default="test",
        # choices=["test", "train", "equivalence_check"],
        help="set algorithm mode",
    )
    parser.add_argument(
        "--expt_name",
        type=str,
        choices=["default", "unitprice", "realprice"],
    )
    parser.add_argument(
        "--use_pretrained_ckpt",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--epi_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--is_waiting",
        action="store_true",
        default=False,
        help="only available if the scenario == 'scim' ",
    )
    parser.add_argument(
        "--demand_pattern",
        type=str,
        default="fixed",
        help="demand generation pattern",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--is_unit_price",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fuel",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    return args


def get_configs(path: Path = None) -> Dict:
    if path is None:
        path = Path(__file__).parent / "config.toml"

    with open(path, "rb") as f:
        cfg = tomli.load(f)

    return cfg


def init_path(
    scenario: str,
    dataset: str,
    algo: str,
    algo_mode: str,
    expt_name: str = "",
    use_pretrained_ckpt: bool = False,
    ecr_is_unit_price: bool = False,
    ecr_fuel: float = 0.0,
    scim_demand_pattern: str = "cosine",
) -> Dict[str, Path]:
    base_path = Path(__file__).parent
    ckpt_path = base_path / "ckpt" / scenario
    data_path = base_path / "data" / scenario / dataset
    res_path = base_path / "results" / scenario / dataset
    img_path = base_path / "results" / scenario
    cplex_respath = Path(f"{expt_name}_{dataset}_{algo}_{algo_mode}_{ecr_fuel}")

    if scenario == "ecr":
        assert expt_name in ["unitprice", "realprice"]
        if use_pretrained_ckpt:
            assert dataset in ["sz", "nyc"]
            assert ecr_is_unit_price is False
            assert ecr_fuel == 0.5
            ckpt_path = ckpt_path / "pretrained"

        res_path = res_path / expt_name
        if algo == "dmfc":
            res_path = res_path / "dmfc"
            # res_path = res_path / "dmfc_test"
        elif algo == "grl":
            if use_pretrained_ckpt:
                res_path = res_path / "grl_pretrained"
            else:
                if ecr_fuel > 0:
                    res_path = res_path / "grl_withfuel"
                else:
                    res_path = res_path / "grl_nofuel"

        ## ckpt path
        fuel_str = f"fuel{ecr_fuel}"
        price_str = "unitprice" if ecr_is_unit_price else "realprice"
        ckpt_path /= f"{dataset}_{algo}_{price_str}_{fuel_str}_ckpt.pth"

    elif scenario == "scim":
        assert expt_name is None
        if use_pretrained_ckpt:
            assert dataset in ["1f2s", "1f3s", "1f10s"]
            ckpt_path = ckpt_path / "pretrained"
        ckpt_path /= f"{dataset}_{algo}_ckpt.pth"
        res_path = res_path / algo / scim_demand_pattern
    else:
        pass

    paths = {
        "ckpt_path": ckpt_path,
        "data_path": data_path,
        "res_path": res_path,
        "img_path": img_path,
        "cplex_respath": cplex_respath,
    }
    return paths


def initialize_components(
    scenario: str,
    dataset: str,
    algo: str,
    mode: str = "test",
    use_pretrained_ckpt: bool = False,
    ecr_is_unit_price: bool = False,
    ecr_fuel: float = 0.0,
    scim_demand_pattern: str = "cosine",
    expt_name: str = None,
    use_cuda: bool = False,
    log_level: int = 0,
    use_wandb: bool = False,
    wandb_key: str = None,
    cplex_path: str = None,
) -> Dict[str, Any]:
    paths = init_path(
        scenario=scenario,
        dataset=dataset,
        algo=algo,
        algo_mode=mode,
        expt_name=expt_name,
        use_pretrained_ckpt=use_pretrained_ckpt,
        ecr_is_unit_price=ecr_is_unit_price,
        ecr_fuel=ecr_fuel,
        scim_demand_pattern=scim_demand_pattern,
    )

    if scenario == "scim":
        if algo == "dmfc":
            algo_instance = SCIMDMFC(mode=mode)
        elif algo == "grl":
            algo_instance = SCIMGRL(
                mode=mode,
                ckpt_path=paths["ckpt_path"],
                dataset=dataset,
                use_cuda=use_cuda,
            )
        else:
            raise ValueError(f"Unknown algo: {algo}")
        agent = SCIMAgent(algo=algo_instance)
        env = SCIMEnv(demand_pattern=scim_demand_pattern)
        data = load_scim_data(dataset)
        graph = SCIMGraph(data=data)
        logger = SCIMLogger(
            scenario="scim",
            dataset=dataset,
            resdir=paths["res_path"],
            log_level=log_level,
            use_wandb=use_wandb,
            wandb_key=wandb_key,
            mode=mode,
            expt_name=expt_name,
        )
    elif scenario == "ecr":
        if algo == "dmfc":
            algo_instance = ECRDMFC(mode=mode)
        elif algo in ["grl", "equivalence_check"]:
            algo_instance = ECRGRL(
                mode=mode,
                dataset=dataset,
                ckpt_path=paths["ckpt_path"],
                use_cuda=use_cuda,
                cplex_path=cplex_path,
                cplex_respath=paths["cplex_respath"],
            )
        else:
            raise ValueError(f"Unknown algo: {algo}")
        agent = ECRAgent(algo=algo_instance)
        if ecr_is_unit_price:
            ecr_fuel = 0.0  # no fuel cost with unit price
        if mode == "train":
            assert expt_name in ["realprice", "unitprice"]  # no noise in training
        env = ECREnv(
            fuel=ecr_fuel,
            expt_name=expt_name,
            cplexpath=cplex_path,
            algo_mode=mode,
            cplex_respath=paths["cplex_respath"],
        )
        graph = ECRGraph(
            data=load_ecr_data(dataset, is_unit_price=ecr_is_unit_price, beta=ecr_fuel),
        )
        logger = ECRLogger(
            scenario="ecr",
            dataset=dataset,
            resdir=paths["res_path"],
            log_level=log_level,
            use_wandb=use_wandb,
            wandb_key=wandb_key,
            mode=mode,
            expt_name=expt_name,
            ecr_fuel=ecr_fuel,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    components = {
        "paths": paths,
        "agent": agent,
        "env": env,
        "graph": graph,
        "logger": logger,
    }

    return components


if __name__ == "__main__":
    args = get_args()
    cfgs = get_configs()
    keys = ["wandb_key", "ibm_cplex"]
    for key in keys:
        setattr(args, key, cfgs[key])

    # fuel init
    if args.expt_name == "unitprice":
        args.is_unit_price = True
        args.fuel = 0.0
    elif args.expt_name == "realprice":
        args.is_unit_price = False
        if args.fuel:
            if args.dataset == "didi20":
                args.fuel = 0.05  # scaling
            else:
                args.fuel = 0.5
        else:
            args.fuel = 0.0

    ## init random seed
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    components = initialize_components(
        scenario=args.scenario,
        dataset=args.dataset,
        algo=args.algo,
        mode=args.algo_mode,
        expt_name=args.expt_name,
        use_pretrained_ckpt=args.use_pretrained_ckpt,
        ecr_is_unit_price=args.is_unit_price,
        ecr_fuel=args.fuel,
        scim_demand_pattern=args.demand_pattern,
        use_cuda=args.use_cuda,
        log_level=args.log_level,
        use_wandb=args.use_wandb,
        wandb_key=args.wandb_key,
        cplex_path=args.ibm_cplex,
    )
    simulator = Simulator()
    simulator.run(
        components["graph"],
        components["agent"],
        components["env"],
        components["logger"],
        args.epi_num,
    )
