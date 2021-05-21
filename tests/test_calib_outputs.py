import numpy as np
from liscal import calibration, utils


def test_phistory_ranked(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_result = dummy_cfg.path_result
    utils.run_cmd("mkdir -p {}".format(path_result))

    pHistory = calibration.read_param_history(path_subcatch)
    pHistory_ranked = calibration.write_ranked_solution(path_result, pHistory)

    ret, out = utils.run_cmd('diff {}/pHistoryWRanks.csv {}/pHistoryWRanks.csv'.format(path_subcatch, path_result))
    print(out)
    assert out == ''
    assert ret == 0


def test_pareto_front(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_result = dummy_cfg.path_result
    utils.run_cmd("mkdir -p {}".format(path_result))
    
    pHistory = calibration.read_param_history(path_subcatch)
    pHistory_ranked = calibration.write_ranked_solution(path_subcatch, pHistory)
    calibration.write_pareto_front(param_ranges, path_result, pHistory_ranked)

    ret, out = utils.run_cmd('diff {}/pareto_front.csv {}/pareto_front.csv'.format(path_subcatch, path_result))
    print(out)
    assert out == ''
    assert ret == 0


class Criteria():

    effmax = np.array([[0.9999384017071802], [0.9999384017071802]])
    effmin = np.array([[0.9999384017071802], [0.9999384017071802]])
    effstd = np.array([[0.0], [0.0]])
    effavg = np.array([[0.9999384017071802], [0.9999384017071802]])

def test_front_history(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    path_result = dummy_cfg.path_result
    utils.run_cmd("mkdir -p {}".format(path_result))

    criteria = Criteria()

    calibration.write_front_history(criteria, path_result, 2)

    ret, out = utils.run_cmd('diff {}/front_history.csv {}/front_history.csv'.format(path_subcatch, path_result))
    print(out)
    assert out == ''
    assert ret == 0
