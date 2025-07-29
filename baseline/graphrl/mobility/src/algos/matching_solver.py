import os
import subprocess
from collections import defaultdict
from pathlib import Path

from loguru import logger

from baseline.graphrl.mobility.src.misc.utils import mat2str


def matching(t, edges, demandAttr, accTuple, CPLEXPATH=None, PATH="", platform="linux"):
    # logger.warning(f"edges: {sorted(edges)}")
    # logger.warning(f"demand: {demandAttr}")
    # logger.warning(f"available: {accTuple}")
    # logger.warning(f"cplex path: {CPLEXPATH}")

    modPath = Path(__file__).parent.parent / "cplex_mod"
    matchingPath = (
        Path(__file__).parents[2] / "saved_files" / "cplex_logs" / "matching" / PATH
    )
    # if not os.path.exists(matchingPath):
    #     os.makedirs(matchingPath)
    matchingPath.mkdir(parents=True, exist_ok=True)

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
    flow = defaultdict(float)
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
    paxAction = {(i, j): flow[i, j] for i, j in sorted(edges) if flow[i, j] > 0}
    return paxAction
