import codecs
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger as log

from baseline.graphrl.supplychain.src.misc.utils import mat2str


def solveLCP(
    env,
    desiredDistrib=None,
    desiredProd=None,
    noiseTuple=None,
    sink=None,
    CPLEXPATH=None,
    res_path="scim",
    root_path="/mnt/array/daga_data/Github/SCIMAI-Gym",
):
    t = env.time
    availableProd = [
        (i, max(env.acc[t - 1][i] + env.arrival_prod[t][i], 0))
        for i in env.scenario.factory
    ]
    desiredShip = [
        (
            env.scenario.warehouse[i],
            int(desiredDistrib[i] * sum([v for i, v in availableProd])),
        )
        for i in range(len(env.scenario.warehouse))
    ]
    desiredProd = [
        (i, max(int(desiredProd[i].item()), 0)) for i in env.scenario.factory
    ]
    storageCapacity = [(i, env.scenario.storage_capacities[i]) for i in env.nodes]
    warehouseStock = [
        (i, int(env.acc[t - 1][i] + env.arrival_flow[t][i]))
        for i in env.scenario.warehouse
    ]
    edgeAttr = [
        (
            i,
            j,
            env.random_graph.edges[(i, j)]["edge_time"],
            env.random_graph.edges[(i, j)]["edge_cost"],
        )
        for i, j in env.random_graph.edges
    ]
    demand = [(i, env.demand[t][i]) for i in env.scenario.warehouse]

    return solve(
        t=t,
        availableProd=availableProd,
        desiredShip=desiredShip,
        desiredProd=desiredProd,
        storageCapacity=storageCapacity,
        warehouseStock=warehouseStock,
        edgeAttr=edgeAttr,
        demand=demand,
        edges=env.G.edges,
        factories=env.scenario.factory,
        res_path=res_path,
        CPLEXPATH=CPLEXPATH,
    )


def solve(
    t,
    availableProd,
    desiredShip,
    desiredProd,
    storageCapacity,
    warehouseStock,
    edgeAttr,
    demand,
    edges,
    factories,
    res_path,
    CPLEXPATH,
):
    # log.warning(f"t: {t}")
    # log.warning(f"availableProd:{availableProd}")
    # log.warning(f"desiredShip:{desiredShip}")
    # log.warning(f"desiredProd:{desiredProd}")
    # log.warning(f"storageCapacity:{storageCapacity}")
    # log.warning(f"warehouseStock:{warehouseStock}")
    # log.warning(f"edgeAttr:{edgeAttr}")
    # log.warning(f"demand:{demand}")
    # log.warning(f"edges:{edges}")
    # log.warning(f"factories:{factories}")
    # log.warning(f"res_path:{res_path}")
    # log.warning(f"CPLEXPATH:{CPLEXPATH}")

    modPath = Path(__file__).parent.parent / "cplex_mod"
    matchingPath = Path(__file__).parents[2] / "saved_files" / "cplex_logs" / res_path

    matchingPath.mkdir(parents=True, exist_ok=True)
    datafile = matchingPath / f"data_{t}.dat"
    resfile = matchingPath / f"res_{t}.dat"
    with open(datafile, "w") as file:
        file.write(f'path="{resfile}";\r\n')
        file.write(f"availableProd={mat2str(availableProd)};\r\n")
        file.write(f"desiredShip={mat2str(desiredShip)};\r\n")
        file.write(f"desiredProd={mat2str(desiredProd)};\r\n")
        file.write(f"storageCapacity={mat2str(storageCapacity)};\r\n")
        file.write(f"warehouseStock={mat2str(warehouseStock)};\r\n")
        file.write(f"demand={mat2str(demand)};\r\n")
        file.write(f"edgeAttr={mat2str(edgeAttr)};\r\n")
    modfile = modPath / "lcp.mod"
    if CPLEXPATH is None:
        CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = matchingPath / f"out_{t}.dat"
    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )
    output_f.close()
    flow = defaultdict(float)
    production = defaultdict(float)
    with codecs.open(resfile, "r", encoding="utf8", errors="ignore") as file:
        for row in file:
            item = (
                row.replace("e)", ")")
                .strip()
                .strip(";")
                .replace("?", "")
                .replace("\x1f", "")
                .replace("\x0f", "")
                .replace("\x7f", "")
                .replace("/", "")
                .replace("O", "")
                .split("=")
            )
            if item[0] == "flow":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(",")
                    flow[int(i), int(j)] = float(
                        f.replace("y6", "")
                        .replace("I0", "")
                        .replace("\x032", "")
                        .replace("C8", "")
                        .replace("C3", "")
                        .replace("c5", "")
                        .replace("#9", "")
                        .replace("c9", "")
                        .replace("\x132", "")
                        .replace("c2", "")
                        .replace("\x138", "")
                        .replace("c2", "")
                        .replace("\x133", "")
                        .replace("\x131", "")
                        .replace("s", "")
                        .replace("#0", "")
                        .replace("c4", "")
                        .replace("\x031", "")
                        .replace("c8", "")
                        .replace("\x037", "")
                        .replace("\x034", "")
                        .replace("s4", "")
                        .replace("S3", "")
                        .replace("\x139", "")
                        .replace("\x138", "")
                        .replace("C4", "")
                        .replace("\x039", "")
                        .replace("S8", "")
                        .replace("\x033", "")
                        .replace("S5", "")
                        .replace("#", "")
                        .replace("\x131", "")
                        .replace("\t6", "")
                        .replace("\x01", "")
                        .replace("i9", "")
                        .replace("y4", "")
                        .replace("a6", "")
                        .replace("y5", "")
                        .replace("\x018", "")
                        .replace("I5", "")
                        .replace("\x11", "")
                        .replace("y2", "")
                        .replace("\x011", "")
                        .replace("y4", "")
                        .replace("y5", "")
                        .replace("a2", "")
                        .replace("i9", "")
                        .replace("i7", "")
                        .replace("\t3", "")
                        .replace("q", "")
                        .replace("I3", "")
                        .replace("A", "")
                        .replace("y5", "")
                        .replace("Q", "")
                        .replace("a3", "")
                        .replace("\x190", "")
                        .replace("\x013", "")
                        .replace("o", "")
                        .replace("`", "")
                        .replace("\x10", "")
                        .replace("P", "")
                        .replace("p", "")
                        .replace("@", "")
                        .replace("M", "")
                        .replace("]", "")
                        .replace("?", "")
                        .replace("\x1f", "")
                        .replace("}", "")
                        .replace("m", "")
                        .replace("\x04", "")
                        .replace("\x0f", "")
                        .replace("\x7f", "")
                        .replace("T", "")
                        .replace("$", "")
                        .replace("t", "")
                        .replace("\x147", "")
                        .replace("\x14", "")
                        .replace("\x046", "")
                        .replace("\x042", "")
                        .replace("/", "")
                        .replace("O", "")
                        .replace("D", "")
                        .replace("d", "")
                        .replace(")", "")
                        .replace("Y", "")
                        .replace("i", "")
                        .replace("\x193", "")
                        .replace("\x192", "")
                        .replace("y5", "")
                        .replace("I2", "")
                        .replace("\t", "")
                        .replace("i2", "")
                        .replace("!", "")
                        .replace("i7", "")
                        .replace("A8", "")
                    )
            if item[0] == "production":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, p = v.split(",")
                    production[int(i)] = float(p)
    ship = {(i, j): flow[i, j] if (i, j) in flow else 0 for i, j in edges}
    # env.G.edges, env.scenario.factory
    prod = {(i): production[i] if (i) in production else 0 for i in factories}
    action = (prod, ship)
    # log.warning(f"prod: {prod}")
    # log.warning(f"ship: {ship}")
    return action
