import os
import subprocess
from collections import defaultdict
from pathlib import Path

from loguru import logger

from baseline.graphrl.mobility.src.misc.utils import mat2str


def solveLCP(env, res_path, desiredAcc, CPLEXPATH):
    t = env.time
    accRLTuple = [(n, int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]
    ## env.G.edges # no self loop
    ## env.edges # with self loop

    logger.warning(f"target_state: {accRLTuple}")

    modPath = Path(__file__).parent.parent / "cplex_mod"
    OPTPath = (
        Path(__file__).parents[2]
        / "saved_files"
        / "cplex_logs"
        / "rebalancing"
        / res_path
    )

    return solve(
        t=t,
        edges=sorted(env.edges),  # with self loop
        s=accTuple,
        target_s=accRLTuple,
        edgeAttr=edgeAttr,
        modPath=modPath,
        OPTPath=OPTPath,
        CPLEXPATH=CPLEXPATH,
    )


def solve(s, target_s, edges, edgeAttr, t, modPath, OPTPath, CPLEXPATH):
    OPTPath.mkdir(parents=True, exist_ok=True)
    datafile = OPTPath / f"data_{t}.dat"
    resfile = OPTPath / f"res_{t}.dat"
    with open(datafile, "w") as file:
        file.write(f'path="{str(resfile)}";\r\n')
        file.write(f"edgeAttr={mat2str(edgeAttr)};\r\n")
        file.write(f"accInitTuple={mat2str(s)};\r\n")
        file.write(f"accRLTuple={mat2str(target_s)};\r\n")
    modfile = modPath / "lcp.mod"
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = OPTPath / f"out_{t}.dat"
    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )
    output_f.close()

    # 3. collect results from file
    optimal_value = None
    flow = defaultdict(float)
    with open(resfile, "r", encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(";").split("=")
            if item[0] == "flow":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(",")
                    flow[int(i), int(j)] = float(f)
            if item[0] == "Optimal_Value":
                optimal_value = float(item[1])

    action = {(i, j): flow[i, j] for i, j in edges if flow[i, j] > 0 and i != j}
    # action = {(i, j): flow for (i, j), flow in action.items() if i != j}
    return action
