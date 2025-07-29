from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

import wandb


class BaseLogger:
    def __init__(
        self,
        scenario: str,
        dataset: str,
        resdir: str | Path,
        use_wandb: bool = False,
        wandb_key: str = "",
        log_level: int = 20,
        mode: str = "test",
        expt_name: str = "",
        ecr_fuel: float = 0.0,
    ):
        self._scenario: str = scenario
        self._dataset: str = dataset
        self._use_wandb: bool = use_wandb
        self._resdir = resdir
        self._mode = mode
        self._logs = defaultdict(list)

        ## wandb init
        if use_wandb:
            wandb.login(key=wandb_key)
            wandb.init(
                project=f"{self._scenario}_{self._dataset}",
                # config=self._args,
            )

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
            level=log_level,
            format=format_string,
            backtrace=True,
            diagnose=True,
        )
        if mode == "test":
            path = Path(__file__).parent.parent / f"{mode}_logs"
            path.mkdir(parents=True, exist_ok=True)
            logger.add(
                path / f"{self._dataset}_{expt_name}_{ecr_fuel}.log",
                level=log_level,
                format=format_string,
                mode="w",
            )

    def log(self, data):
        raise NotImplementedError

    def log_episode(self):
        raise NotImplementedError

    def clear_logs(self):
        self._logs = defaultdict(list)

    def finish(self):
        if self._use_wandb:
            wandb.finish()


class SCIMLogger(BaseLogger):
    def log(self, data):
        self._logs["step_rewards"].append(data["reward"])
        self._logs["demands"].append(data["demand_t"])

        self._logs["stocks"].append(data["stock_level"])
        self._logs["storage_cost"].append(data["storage_cost"])
        self._logs["overstock_cnt"].append(data["overstock_cnt"])

        self._logs["inflows"].append(data["node_inflows"])
        self._logs["outflows"].append(data["outflows"])
        self._logs["waiting_cnt"].append(data["waiting_cnt"])

        self._logs["prods"].append(data["production"])
        self._logs["trans_cost"].append(data["trans_cost"])
        # self._logs.append(data)

        if self._use_wandb:
            wandb_keys = [
                "revenue",
                "waiting_panelty",
                "trans_cost",
                "prod_cost",
                "storage_cost",
                "overstock_panelty",
                "step_reward",
            ]
            demand_t = data.get("demand_t", None)
            demand_dict = {f"demand_{i}": demand_t[i] for i in range(len(demand_t))}
            log_info = {key: data[key] for key in wandb_keys if key in data}
            log_info.update(demand_dict)
            wandb.log(log_info)

    def log_episode(self, t):
        if self._mode == "train":
            return

        Path.mkdir(self._resdir, exist_ok=True, parents=True)

        with open(self._resdir / "step_rewards.csv", "a") as f:
            step_rewards = np.array(self._logs["step_rewards"])[None, :]
            np.savetxt(f, step_rewards, delimiter=",", fmt="%.10f")
        with open(self._resdir / "rewards.csv", "a") as f:
            np.savetxt(f, [step_rewards.sum()], delimiter=",", fmt="%.10f")
        with open(self._resdir / "demands.csv", "a") as f:
            demands = np.array(self._logs["demands"]).flatten()[None, :]
            np.savetxt(f, demands, delimiter=",", fmt="%.10f")

        with open(self._resdir / "inflows.csv", "a") as f:
            inflows = np.array(self._logs["inflows"]).flatten()[None, :]
            np.savetxt(f, inflows, delimiter=",", fmt="%.10f")
        with open(self._resdir / "outflows.csv", "a") as f:
            outflows = np.array(self._logs["outflows"]).flatten()[None, :]
            np.savetxt(f, outflows, delimiter=",", fmt="%.10f")

        with open(self._resdir / "prods.csv", "a") as f:
            prods = np.array(self._logs["prods"])[None, :]
            np.savetxt(f, prods, delimiter=",", fmt="%.10f")

        with open(self._resdir / "stocks.csv", "a") as f:
            stock_level = np.array(self._logs["stocks"]).flatten()[None, :]
            np.savetxt(f, stock_level, delimiter=",", fmt="%.10f")
        with open(self._resdir / "storage_cost.csv", "a") as f:
            storage_cost = np.array(self._logs["storage_cost"])[None, :]
            np.savetxt(f, storage_cost, delimiter=",", fmt="%.10f")
        with open(self._resdir / "overstocks.csv", "a") as f:
            overstocks = np.array(self._logs["overstock_cnt"])[None, :]
            np.savetxt(f, overstocks, delimiter=",", fmt="%.10f")

        with open(self._resdir / "waitings.csv", "a") as f:
            waitings = np.array(self._logs["waiting_cnt"])[None, :]
            np.savetxt(f, waitings, delimiter=",", fmt="%.10f")

        with open(self._resdir / "trans_cost.csv", "a") as f:
            trans_cost = np.array(self._logs["trans_cost"])[None, :]
            np.savetxt(f, trans_cost, delimiter=",", fmt="%.10f")


class ECRLogger(BaseLogger):
    def log(self, data):
        self._logs["step_reward"].append(data["reward"])
        self._logs["demands"].append(data["demands"])
        self._logs["satisfied"].append(data["satisfied"])
        self._logs["step_completion_rate"].append(data["step_fulfillment_rate"])
        self._logs["e_fuel_cost"].append(data["empty_car_fuel_cost"])
        self._logs["f_fuel_cost"].append(data["full_car_fuel_cost"])
        self._logs["cum_completion_rate"].append(data["cum_fulfillment_rate"])

        if self._use_wandb and self._mode == "test":
            info = {
                "step_demand": np.sum(data["demands"]),
                "step_served_demand": np.sum(data["satisfied"]),
                "step_completion_rate": data["step_fulfillment_rate"],
                "cum_completion_rate": data["cum_fulfillment_rate"],
                "e_fuel_cost": data["empty_car_fuel_cost"],
                "f_fuel_cost": data["full_car_fuel_cost"],
            }
            wandb.log(info)

    def log_episode(self, t):
        if self._mode == "train" and self._use_wandb:
            info = {
                "demands": np.sum(self._logs["demands"]),
                "served_demand": np.sum(self._logs["satisfied"]),
                "rebalancing_cost": np.sum(self._logs["e_fuel_cost"]),
                "reward": np.sum(self._logs["step_reward"]),
            }
            wandb.log(info, step=t)
            return

        Path.mkdir(self._resdir, exist_ok=True, parents=True)
        with open(self._resdir / "rewards.csv", "a") as f:
            rewards = np.array(self._logs["step_reward"]).sum()
            np.savetxt(f, [rewards], delimiter=",", fmt="%.10f")
        demands = np.array(self._logs["demands"])  # (200, R, R)
        with open(self._resdir / "demands.csv", "a") as f:
            np.savetxt(f, demands.flatten()[None, :], delimiter=",", fmt="%.10f")
        satisfied = np.array(self._logs["satisfied"])
        with open(self._resdir / "satisfied.csv", "a") as f:
            np.savetxt(f, satisfied.flatten()[None, :], delimiter=",", fmt="%.10f")
        with open(self._resdir / "demand.csv", "a") as f:
            np.savetxt(f, [demands.sum()], delimiter=",", fmt="%.10f")
        with open(self._resdir / "step_completion_rate.csv", "a") as f:
            completion_rate = np.array(self._logs["step_completion_rate"])[None, :]
            np.savetxt(f, completion_rate, delimiter=",", fmt="%.10f")
        with open(self._resdir / "completion_rate.csv", "a") as f:
            completion_rate = np.array(self._logs["cum_completion_rate"])[None, :]
            np.savetxt(f, completion_rate, delimiter=",", fmt="%.10f")
