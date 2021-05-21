import subprocess
import pandas


def read_tss(tss_file):
    df = pandas.read_csv(tss_file, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
    return df


def run_cmd(cmd):
  res = subprocess.run(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if res.returncode == 0:
    out = res.stdout[0:-1]
  else:
    Exception(res.stderr)
    out = res.stderr
  return res.returncode, out
