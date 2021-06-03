import numpy as np
import xarray as xr
from liscal import subcatchment, objective, utils


def test_phistory_ranked(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking pHistory file')

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, None, initialise=False)
    obj = objective.ObjectiveDischarge(dummy_cfg, subcatch, read_observations=False)

    pHistory = obj.read_param_history()
    pHistory_ranked = obj.write_ranked_solution(pHistory, path_out=path_out)

    cmd = 'diff {}/pHistoryWRanks.csv {}/pHistoryWRanks.csv'.format(path_subcatch, path_out)
    ret, out = utils.run_cmd(cmd)
    print(out)
    assert out == ''
    assert ret == 0


def test_pareto_front(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking pareto_front file')

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, None, initialise=False)
    obj = objective.ObjectiveDischarge(dummy_cfg, subcatch, read_observations=False)

    pHistory = obj.read_param_history()
    pHistory_ranked = obj.write_ranked_solution(pHistory, path_out=path_out)
    obj.write_pareto_front(pHistory_ranked, path_out=path_out)

    cmd = 'diff {}/pareto_front.csv {}/pareto_front.csv'.format(path_subcatch, path_out)
    ret, out = utils.run_cmd(cmd)
    print(out)
    print(cmd)
    assert out == ''
    assert ret == 0
