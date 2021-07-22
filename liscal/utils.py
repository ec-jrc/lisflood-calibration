import subprocess
import pandas as pd


def read_tss(tss_file, skiprows=4):
    df = pd.read_csv(tss_file, sep=r"\s+", index_col=0, skiprows=skiprows, header=None, skipinitialspace=True)
    return df


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        out = res.stdout[0:-1]
    else:
        Exception(res.stderr)
        out = res.stdout[0:-1]
        out += res.stderr
    return res.returncode, out
