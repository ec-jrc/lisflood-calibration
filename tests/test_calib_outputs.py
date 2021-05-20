import subprocess
from liscal import calibration


def run_cmd(cmd):
  res = subprocess.run(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if res.returncode == 0:
    out = res.stdout[0:-1]
  else:
    Exception(res.stderr)
    out = res.stderr
  return res.returncode, out


def test_phistory_ranked(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_result = dummy_cfg.path_result
    pHistory = calibration.read_param_history(path_subcatch)
    pHistory_ranked = calibration.write_ranked_solution(path_result, pHistory)

    run_cmd("mkdir -p {}".format(path_result))
    ret, out = run_cmd('diff {}/pHistoryWRanks.csv {}/pHistoryWRanks.csv'.format(path_subcatch, path_result))
    print(out)
    assert out == ''
    assert ret == 0


def test_pareto_front(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_result = dummy_cfg.path_result

    run_cmd("mkdir -p {}".format(path_result))
    pHistory = calibration.read_param_history(path_subcatch)
    pHistory_ranked = calibration.write_ranked_solution(path_subcatch, pHistory)
    calibration.write_pareto_front(param_ranges, path_result, pHistory_ranked)

    ret, out = run_cmd('diff {}/pareto_front.csv {}/pareto_front.csv'.format(path_subcatch, path_result))
    print(out)
    assert out == ''
    assert ret == 0
