import numpy as np
import xarray as xr
from liscal import calibration, utils


def test_front_history(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    path_out = dummy_cfg.path_out
    deap_param = dummy_cfg.deap_param

    print('checking front_history file')

    criteria = calibration.Criteria(deap_param)
    criteria.eff_KGE['max'] = np.array([[0.9999384017071802], [0.9999384017071802]])
    criteria.eff_KGE['min'] = np.array([[0.9999384017071802], [0.9999384017071802]])
    criteria.eff_KGE['std'] = np.array([[0.0], [0.0]])
    criteria.eff_KGE['avg'] = np.array([[0.9999384017071802], [0.9999384017071802]])

    criteria.write_front_history(path_out, 2)

    cmd = 'diff {}/front_history.csv {}/front_history.csv'.format(path_subcatch, path_out)
    ret, out = utils.run_cmd(cmd)
    print(out)
    assert out == ''
    assert ret == 0


def test_termination_gen(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    deap_param = dummy_cfg.deap_param

    criteria = calibration.Criteria(deap_param)
    criteria.gen_offset = 1

    assert criteria.conditions['maxGen']is False
    assert criteria.conditions['StallFit'] is False

    gen = criteria.max_gen

    criteria.check_termination_conditions(gen)

    assert criteria.conditions['maxGen'] is True
    assert criteria.conditions['StallFit'] is False


def test_termination_stall(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    deap_param = dummy_cfg.deap_param

    criteria = calibration.Criteria(deap_param)
    print(criteria.eff['max'])
    criteria.gen_offset = 1

    assert criteria.conditions['maxGen'] is False
    assert criteria.conditions['StallFit'] is False

    gen = 1
    criteria.max_gen = 2
    criteria.eff_KGE['max'] = np.array([[0.991], [0.991]])

    criteria.check_termination_conditions(gen)

    assert criteria.conditions['maxGen']is False
    assert criteria.conditions['StallFit'] is True


def test_update(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    deap_param = dummy_cfg.deap_param

    criteria = calibration.Criteria(deap_param)

    halloffame = []
    ds1 = xr.Dataset()
    ds1['fitness'] = xr.DataArray(np.array([0.1]), dims=['x'])
    halloffame.append(ds1)
    ds2 = xr.Dataset()
    ds2['fitness'] = xr.DataArray(np.array([0.2]), dims=['x'])
    halloffame.append(ds2)

    gen = 1
    criteria.update_statistics(gen, halloffame)

    print(criteria.eff['min'][1, 0])
    assert criteria.eff['min'][1, 0] == 0.1
    print(criteria.eff['max'][1, 0])
    assert criteria.eff['max'][1, 0] == 0.2
    print(criteria.eff['std'][1, 0])
    assert criteria.eff['std'][1, 0] == 0.05
    print(criteria.eff['avg'][1, 0])
    assert np.abs(criteria.eff['avg'][1, 0] - 0.15) < 1e-8
