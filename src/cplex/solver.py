import codecs
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


def scim_lcp_solver(
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

    modfile = Path(__file__).parent / "scim_lcp.mod"
    OPTPath = Path(__file__).parent / "logs" / "scim_lcp" / res_path
    OPTPath.mkdir(parents=True, exist_ok=True)
    
    datafile = OPTPath / f"data_{t}.dat"
    resfile = OPTPath / f"res_{t}.dat"
    out_file = OPTPath / f"out_{t}.dat"
    
    with open(datafile, "w") as file:
        resfile = str(resfile).replace("\\", "/")
        file.write(f'path="{resfile}";\r\n')
        file.write(f"availableProd={mat2str(availableProd)};\r\n")
        file.write(f"desiredShip={mat2str(desiredShip)};\r\n")
        file.write(f"desiredProd={mat2str(desiredProd)};\r\n")
        file.write(f"storageCapacity={mat2str(storageCapacity)};\r\n")
        file.write(f"warehouseStock={mat2str(warehouseStock)};\r\n")
        file.write(f"demand={mat2str(demand)};\r\n")
        file.write(f"edgeAttr={mat2str(edgeAttr)};\r\n")

    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    
    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [Path(CPLEXPATH) / "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )

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
    prod = {(i): production[i] if (i) in production else 0 for i in factories}
    action = (prod, ship)
    return action

