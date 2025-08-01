"""
DVR Environment
-----------------------------------------
This file contains the specifications for the DVR system simulator. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import json
import os
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger as log

from src.algos.matching_solver import matching
from src.misc.utils import fraction_allocation, mat2str


class DVR:
    # initialization
    def __init__(self, scenario, beta=0.2):
        self.scenario = deepcopy(scenario)
        self.G = scenario.G
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0
        self.tf = scenario.tf
        self.demand = defaultdict(dict)
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G)
        for i in self.region:
            self.depDemand[i] = defaultdict(int)
            self.arrDemand[i] = defaultdict(int)

        self.price = defaultdict(dict)
        for i, j, t, d, p in scenario.tripAttr:
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t + self.demandTime[i, j][t]] += d
        self.acc = defaultdict(
            dict
        )  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(
            dict
        )  # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(
            dict
        )  # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(
            dict
        )  # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [
            len(self.G.out_edges(n)) + 1 for n in self.region
        ]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.rebTime[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(int)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(int)
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]["accInit"]
            self.dacc[n] = defaultdict(int)
        self.beta = beta * scenario.tstep
        t = self.time
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(int)

        self.N = len(self.region)  # total number of cells

        # add the initialization of info here
        self.info = dict.fromkeys(
            ["revenue", "served_demand", "rebalancing_cost", "operating_cost"], 0
        )
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def matching(self, CPLEXPATH=None, PATH="", platform="linux"):
        t = self.time
        demandAttr = [
            (i, j, self.demand[i, j][t], self.price[i, j][t])
            for i, j in self.demand
            if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3
        ]
        accTuple = [(n, self.acc[n][t + 1]) for n in self.acc]

        # log.info(f"matching_avail: {np.array([self.acc[n][t + 1] for n in self.acc])}")
        # log.info(f"demand_sum: {sum([self.demand[i, j][t] for i, j in self.demand])}")
        # log.info(f"demand: {sorted(demandAttr, key=lambda x: (x[0], x[1]))}")

        modPath = Path(__file__).parent.parent / "cplex_mod"
        matchingPath = (
            Path(__file__).parents[2] / "saved_files" / "cplex_logs" / "matching" / PATH
        )
        if not os.path.exists(matchingPath):
            os.makedirs(matchingPath)
        datafile = matchingPath / f"data_{t}.dat"
        resfile = matchingPath / f"res_{t}.dat"
        with open(datafile, "w") as file:
            file.write(f'path="{str(resfile)}";\r\n')
            file.write(f"demandAttr={mat2str(demandAttr)};\r\n")
            file.write(f"accInitTuple={mat2str(accTuple)};\r\n")
        modfile = modPath / "matching.mod"
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        if platform == "mac":
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
            my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file = matchingPath / f"out_{t}.dat"
        with open(out_file, "w") as output_f:
            subprocess.check_call(
                [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
            )
        output_f.close()
        flow = defaultdict(int)
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(",")
                        flow[int(i), int(j)] = float(f)
                if item[0] == "Optimal_Value":
                    optimal_value = float(item[1])
        # paxAction = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        paxAction = {
            (i, j): int(flow[i, j]) for i, j in sorted(self.edges) if flow[i, j] > 0
        }
        return paxAction

    # pax step
    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH="", platform="linux"):
        t = self.time
        log.warning(f"step: {self.time}")
        self.reward = 0
        for i in self.region:
            self.acc[i][t + 1] = self.acc[i][t]
        self.info["demand"] = sum(
            [self.demand[i, j][t] for i, j in self.demand if t in self.demand[i, j]]
        )

        self.info["served_demand"] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info["revenue"] = 0
        self.info["rebalancing_cost"] = 0
        if paxAction is None:
            # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            t = self.time
            demandAttr = [
                (i, j, self.demand[i, j][t], self.price[i, j][t])
                for i, j in self.demand
                if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3
            ]
            accTuple = [(n, self.acc[n][t + 1]) for n in self.acc]
            log.warning(f"demand.sum: {sum([d for _, _, d, _ in demandAttr])}")
            log.warning(
                f"demand, avail: {np.array([self.acc[n][t + 1] for n in self.acc], dtype=int)}"
            )
            paxAction = matching(
                t,
                self.edges,
                demandAttr,
                accTuple,
                CPLEXPATH=CPLEXPATH,
                PATH=PATH,
                platform=platform,
            )
            # paxAction = self.matching(CPLEXPATH=CPLEXPATH, PATH=PATH, platform=platform)

        self.paxAction = paxAction
        log.warning(f"matching: {paxAction}")
        for i, j in paxAction.keys():
            assert paxAction[i, j] < self.acc[i][t + 1] + 1e-3
            self.paxAction[i, j] = min(self.acc[i][t + 1], paxAction[i, j])
            self.servedDemand[i, j][t] = self.paxAction[i, j]
            self.paxFlow[i, j][t + self.demandTime[i, j][t]] = self.paxAction[i, j]
            self.info["operating_cost"] += (
                self.demandTime[i, j][t] * self.beta * self.paxAction[i, j]
            )
            self.acc[i][t + 1] -= self.paxAction[i, j]
            self.info["served_demand"] += self.servedDemand[i, j][t]
            self.dacc[j][t + self.demandTime[i, j][t]] += self.paxFlow[i, j][
                t + self.demandTime[i, j][t]
            ]
            self.reward += self.paxAction[i, j] * (
                self.price[i, j][t] - self.demandTime[i, j][t] * self.beta
            )
            self.info["revenue"] += self.paxAction[i, j] * (self.price[i, j][t])

        self.obs = (
            self.acc,
            self.time,
            self.dacc,
            self.demand,
        )  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        return self.obs, max(0, self.reward), done, self.info

    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.rebAction = rebAction

        # log.info(
        #     f"reb_avail_car: {np.array([self.acc[n][t + 1] for n in self.acc], dtype=int)}"
        # )

        for i, j in rebAction.keys():
            if i == j or (i, j) not in self.G.edges:
                # i == j, no rebalancing needed
                # (i, j) not in self.G.edges, no edge between i and j
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[(i, j)] = min(self.acc[i][t + 1], rebAction[(i, j)])
            self.rebFlow[i, j][t + self.rebTime[i, j][t]] = self.rebAction[(i, j)]
            self.acc[i][t + 1] -= self.rebAction[(i, j)]
            self.dacc[j][t + self.rebTime[i, j][t]] += self.rebFlow[i, j][
                t + self.rebTime[i, j][t]
            ]
            self.info["rebalancing_cost"] += (
                self.rebTime[i, j][t] * self.beta * self.rebAction[(i, j)]
            )
            self.info["operating_cost"] += (
                self.rebTime[i, j][t] * self.beta * self.rebAction[(i, j)]
            )
            self.reward -= self.rebTime[i, j][t] * self.beta * self.rebAction[(i, j)]
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing
        inflows = 0
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t + 1] += self.rebFlow[i, j][t]
                inflows += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t + 1] += self.paxFlow[i, j][t]
                inflows += self.paxFlow[i, j][t]
                # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion
        # log.warning(f"inflows: {inflows}")

        self.time += 1
        self.obs = (
            self.acc,
            self.time,
            self.dacc,
            self.demand,
        )  # use self.time to index the next time step
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.rebTime[i, j][self.time]
        done = self.tf == t + 1  # if the episode is completed
        # log.info(f"rebreward: {self.reward}")
        return self.obs, self.reward, done, self.info

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        for i, j, t, d, p in tripAttr:
            # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(int)
            self.paxFlow[i, j] = defaultdict(int)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]["accInit"]
            self.dacc[n] = defaultdict(int)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(int)
        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs


class Scenario:
    def __init__(
        self,
        N1=2,
        N2=4,
        tf=60,
        sd=None,
        ninit=5,
        tripAttr=None,
        demand_input=None,
        demand_ratio=None,
        trip_length_preference=0.25,
        grid_travel_time=1,
        fix_price=True,
        alpha=0.2,
        json_file=None,
        json_hr=9,
        json_tstep=2,
        varying_time=False,
        json_regions=None,
    ):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        if json_file == None:
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1 * N2)
            self.G = self.G.to_directed()
            self.demandTime = dict()
            self.rebTime = dict()
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
            for i, j in self.edges:
                self.demandTime[i, j] = defaultdict(
                    lambda: (abs(i // N1 - j // N1) + abs(i % N1 - j % N1))
                    * grid_travel_time
                )
                self.rebTime[i, j] = defaultdict(
                    lambda: (abs(i // N1 - j // N1) + abs(i % N1 - j % N1))
                    * grid_travel_time
                )

            for n in self.G.nodes:
                self.G.nodes[n]["accInit"] = int(ninit)
            self.tf = tf
            self.demand_ratio = defaultdict(list)

            if demand_ratio == None or type(demand_ratio) == list:
                for i, j in self.edges:
                    if type(demand_ratio) == list:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(0, tf + 1, tf / (len(demand_ratio) - 1)),
                                    demand_ratio,
                                )
                            )
                            + [demand_ratio[-1]] * tf
                        )
                    else:
                        self.demand_ratio[i, j] = [1] * (tf + tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(
                                        0, tf + 1, tf / (len(demand_ratio[i, j]) - 1)
                                    ),
                                    demand_ratio[i, j],
                                )
                            )
                            + [1] * tf
                        )
                    else:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(
                                        0,
                                        tf + 1,
                                        tf / (len(demand_ratio["default"]) - 1),
                                    ),
                                    demand_ratio["default"],
                                )
                            )
                            + [1] * tf
                        )
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand() * 2 + 1) * (
                        self.demandTime[i, j][0] + 1
                    )
            if tripAttr != None:  # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand()  # randomly generated demand

        else:
            self.varying_time = varying_time
            self.is_json = True
            with open(json_file, "r") as file:
                data = json.load(file)
            self.tstep = json_tstep
            self.N1 = data["nlat"]
            self.N2 = data["nlon"]
            self.demand_input = defaultdict(dict)
            self.json_regions = json_regions

            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif "region" in data:
                self.G = nx.complete_graph(data["region"])
            else:
                self.G = nx.DiGraph()
                edge_list = [(edge["i"], edge["j"]) for edge in data["topology_graph"]]
                reverse_edges = [(j, i) for (i, j) in edge_list]
                edge_list = list(set(reverse_edges + edge_list))
                self.G.add_edges_from(edge_list)
            self.G = self.G.to_directed()
            self.p = defaultdict(dict)
            self.alpha = 0
            self.demandTime = defaultdict(dict)
            self.rebTime = defaultdict(dict)
            self.json_start = json_hr * 60
            self.tf = tf
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(int)
                self.rebTime[i, j] = 1

            for item in data["demand"]:
                t, o, d, v, tt, p = (
                    item["time_stamp"],
                    item["origin"],
                    item["destination"],
                    item["demand"],
                    item["travel_time"],
                    item["price"],
                )
                if json_regions != None and (
                    o not in json_regions or d not in json_regions
                ):
                    continue
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = (
                        defaultdict(float),
                        defaultdict(float),
                        defaultdict(float),
                    )

                self.demand_input[o, d][(t - self.json_start) // json_tstep] += (
                    v * demand_ratio
                )
                self.p[o, d][(t - self.json_start) // json_tstep] += (
                    p * v * demand_ratio
                )
                self.demandTime[o, d][(t - self.json_start) // json_tstep] += (
                    tt * v * demand_ratio / json_tstep
                )

            for o, d in self.edges:
                for t in range(0, tf * 2):
                    if t in self.demand_input[o, d]:
                        self.p[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] = max(
                            int(round(self.demandTime[o, d][t])), 1
                        )
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demandTime[o, d][t] = 0

            for item in data["rebTime"]:
                hr, o, d, rt = (
                    item["time_stamp"],
                    item["origin"],
                    item["destination"],
                    item["reb_time"],
                )
                if json_regions != None and (
                    o not in json_regions or d not in json_regions
                ):
                    continue
                if varying_time:
                    t0 = int((hr * 60 - self.json_start) // json_tstep)
                    t1 = int((hr * 60 + 60 - self.json_start) // json_tstep)
                    for t in range(t0, t1):
                        self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, tf + 1):
                            self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)

            for item in data["totalAcc"]:
                hr, acc = item["hour"], item["acc"]
                if hr == json_hr + int(round(json_tstep / 2 * tf / 60)):
                    for n in self.G.nodes:
                        self.G.nodes[n]["accInit"] = int(acc / len(self.G))
            self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(self.demand_input[i, j][t])
                        price[i, j][t] = self.p[i, j][t]
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:
            self.static_demand = dict()
            region_rand = np.random.rand(len(self.G)) * self.alpha * 2 + 1 - self.alpha
            if type(self.demand_input) in [float, int, list, np.array]:
                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:
                    self.region_demand = region_rand * np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    prob = np.array(
                        [
                            np.math.exp(
                                -self.rebTime[i, j][0] * self.trip_length_preference
                            )
                            for j in J
                        ]
                    )
                    prob = prob / sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i, J[idx]] = (
                            self.region_demand[i] * prob[idx]
                        )
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = (
                        self.demand_input[i, j]
                        if (i, j) in self.demand_input
                        else self.demand_input["default"]
                    )

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception(
                    "demand_input should be number, array-like, or dictionary-like values"
                )

            # generating demand and prices
            if self.fix_price:
                p = self.p
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(
                        self.static_demand[i, j] * self.demand_ratio[i, j][t]
                    )
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = (
                            min(3, np.random.exponential(2) + 1)
                            * self.demandTime[i, j][t]
                        )
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr


class MyScenario(Scenario):
    def __init__(
        self,
        T: int = 200,
        delta: int = 15,
        start_hour: int = 13,
        dataset: str = "",
        is_unit_price: bool = False,
    ):
        # load data from file
        assert dataset in ["nyc", "sz", "didi9", "didi20"]
        path = Path(__file__).parents[5] / "data" / "ecr" / dataset
        # path = Path(__file__).parents[2] / "data" / dataset
        mu_e = np.load(path / "mu_e.npy")
        mu_f = np.load(path / "mu_f.npy")
        plam = np.load(path / "plam.npy")
        price = np.load(path / "prices.npy")
        adjacency = np.load(path / "adjacency.npy")

        with open(path / "data.json", "r") as f:
            data = json.load(f)
        C = data["car_num"]
        tstep = data["tstep"]

        start = start_hour * 6
        L = T + delta  # episode length (T) + redundant data (delta)
        R = adjacency.shape[0]
        repeat = (L + start) // plam.shape[0] + 1

        plam = np.tile(plam, (repeat, 1, 1))[start : start + L]
        # log.critical(f"plam: {plam.sum()}")
        # log.critical(f"plam[0].sum: {plam[0].sum()}")
        price = np.tile(price, (repeat, 1, 1))[start : start + L]

        self.p_lam = plam
        self.price = price
        self.is_unit_price = is_unit_price
        # log.critical(is_unit_price)
        if is_unit_price:
            self.price = np.ones_like(self.price)
            # self.price[np.argwhere(self.price > 0)] = 1
            # log.critical(self.price[0].sum())

        tt_f = np.maximum(np.round(1 / mu_f), 1).astype(int)
        tt_e = np.maximum(np.round(1 / mu_e), 1).astype(int)

        if np.ndim(tt_e) == 2:
            tt_e = np.tile(tt_e, (len(plam), 1, 1))

        if np.ndim(tt_f) == 2:
            tt_f = np.tile(tt_f, (len(plam), 1, 1))

        if adjacency is None:
            self.G = nx.complete_graph(R)
        else:
            self.G = nx.from_numpy_array(adjacency)
        self.G = self.G.to_directed()
        self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
        self.edges = sorted(self.edges)

        self.tf = T  ### episode length

        self.demandTime = defaultdict(dict)
        self.rebTime = defaultdict(dict)
        self.demand_input = defaultdict(dict)
        self.p = defaultdict(dict)

        for o, d in self.edges:
            self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = (
                defaultdict(float),
                defaultdict(float),
                defaultdict(float),
            )
            for t in range(len(plam)):
                self.p[o, d][t] = price[t, o, d]
                self.demandTime[o, d][t] = tt_f[t, o, d]
                self.demand_input[o, d][t] = plam[t, o, d]
            for t in range(T + 1):
                self.rebTime[o, d][t] = tt_e[t, o, d]

        self.tstep = tstep

        lam_0 = np.sum(plam[0], axis=1)
        frac = lam_0 / np.sum(lam_0)
        # log.info(frac)
        init = fraction_allocation(C, R, frac)
        for n in self.G.nodes:
            self.G.nodes[n]["accInit"] = init[n]

        self.tripAttr = None

    @property
    def adjacency(self):
        return nx.to_numpy_array(self.G)

    def get_random_demand(self, reset=False):
        # log.debug(f"demand_gen, reset: {reset}")
        demand = np.random.poisson(self.p_lam)
        T, R, _ = demand.shape
        self.demand = demand
        # log.info(f"dtype of demand: {demand.dtype}")
        # log.info(f"shape of demand: {demand.shape}")
        # log.info(f"demand.sum: {np.sum(demand)}")
        # log.info(f"demand[0].sum: {np.sum(demand[0])}")
        # log.info(f"demand[1].sum: {np.sum(demand[1])}")
        # log.info(f"adjacency: {self.adjacency.astype(int)}")
        # log.info(f"n_edge: {len(self.edges)}")
        # log.info(f"edge_list: {self.edges}")
        tripAttr = []
        for t in range(T):
            for i, j in self.edges:
                tripAttr.append((i, j, t, demand[t, i, j], self.price[t, i, j]))
        return tripAttr


class MyDVR(DVR):
    def __init__(self, scenario: MyScenario, beta=0.2):
        self.scenario = deepcopy(scenario)

        self.G = scenario.G
        self.region = list(self.G)
        self.nregion = len(scenario.G)
        self.N = len(self.region)  # total number of cells

        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n)) + 1 for n in self.region]

        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0
        self.tf = scenario.tf
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.rebTime[i, j][self.time]

        self.demand = defaultdict(lambda: defaultdict())
        self.depDemand = defaultdict(lambda: defaultdict(int))
        self.arrDemand = defaultdict(lambda: defaultdict(int))

        self.price = defaultdict(lambda: defaultdict())
        self.acc = defaultdict(lambda: defaultdict(int))
        self.dacc = defaultdict(lambda: defaultdict(int))
        self.rebFlow = defaultdict(lambda: defaultdict(int))
        self.paxFlow = defaultdict(lambda: defaultdict(int))
        self.servedDemand = defaultdict(lambda: defaultdict(int))

        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]["accInit"]

        self.beta = beta * scenario.tstep
        if self.scenario.is_unit_price:
            self.beta = 0

        self.info = dict.fromkeys(
            ["revenue", "served_demand", "rebalancing_cost", "operating_cost"], 0
        )
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)
