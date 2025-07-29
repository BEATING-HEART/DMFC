import os
import subprocess
from collections import defaultdict
from pathlib import Path


def mat2str(mat):
    return (
        str(mat)
        .replace("'", '"')
        .replace("(", "<")
        .replace(")", ">")
        .replace("[", "{")
        .replace("]", "}")
    )


def ecr_lcp_solver(t, s, target_s, edges, edgeAttr, CPLEXPATH, res_path=""):
    modfile = Path(__file__).parent / "ecr_lcp.mod"
    OPTPath = Path(__file__).parent / "logs" / "ecr_lcp" / res_path
    OPTPath.mkdir(parents=True, exist_ok=True)

    datafile = OPTPath / f"data_{t}.dat"
    resfile = OPTPath / f"res_{t}.dat"
    out_file = OPTPath / f"out_{t}.dat"

    with open(datafile, "w") as file:
        resfile = str(resfile).replace("\\", "/")
        file.write(f'path="{str(resfile)}";\r\n')
        file.write(f"edgeAttr={mat2str(edgeAttr)};\r\n")
        file.write(f"accInitTuple={mat2str(s)};\r\n")
        file.write(f"accRLTuple={mat2str(target_s)};\r\n")

    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH

    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [Path(CPLEXPATH) / "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )

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
    return action


def ecr_matching_solver(t, edges, demandAttr, accTuple, CPLEXPATH=None, res_path=""):
    modfile = Path(__file__).parent / "ecr_matching.mod"
    matchingPath = Path(__file__).parent / "logs" / "ecr_matching" / res_path
    matchingPath.mkdir(parents=True, exist_ok=True)

    datafile = matchingPath / f"data_{t}.dat"
    resfile = matchingPath / f"res_{t}.dat"
    out_file = matchingPath / f"out_{t}.dat"

    with open(datafile, "w") as file:
        resfile = str(resfile).replace("\\", "/")
        file.write(f'path="{str(resfile)}";\r\n')
        file.write(f"demandAttr={mat2str(demandAttr)};\r\n")
        file.write(f"accInitTuple={mat2str(accTuple)};\r\n")

    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH

    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [Path(CPLEXPATH) / "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )

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

    paxAction = {(i, j): flow[i, j] for i, j in sorted(edges) if flow[i, j] > 0}
    return paxAction
