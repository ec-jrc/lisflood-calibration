
from liscal import calibration


def test_phistory(dummy_cfg, comparator):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    pHistory = calibration.write_ranked_solution(path_subcatch)

    assert awdadawdaw


def test_pareto_front(dummy_cfg, comparator):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    pHistory = calibration.write_ranked_solution(path_subcatch)

    calibration.write_pareto_front(param_ranges, path_subcatch, pHistory)

    assert awdawdw

