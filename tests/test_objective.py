import os
import gzip
import pytest
from datetime import datetime
import numpy as np
import xarray as xr
from liscal import subcatchment, objective, utils, hydro_stats


def test_phistory_ranked(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking pHistory file')

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data={}, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)

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

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data={}, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)

    pHistory = obj.read_param_history()
    pHistory_ranked = obj.write_ranked_solution(pHistory, path_out=path_out)
    obj.write_pareto_front(pHistory_ranked, path_out=path_out)

    cmd = 'diff {}/pareto_front.csv {}/pareto_front.csv'.format(path_subcatch, path_out)
    ret, out = utils.run_cmd(cmd)
    print(out)
    print(cmd)
    assert out == ''
    assert ret == 0


def gzip_file(file_path):

    infile = file_path+'.gz'
    outfile = file_path
    with gzip.open(infile, 'rb') as f_in:
        content = f_in.read()
    with open(outfile, 'wb') as f_out:
        f_out.write(content)


def test_kge_synthetic(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking kge function')

    cal_start = '31/12/2016 06:00'
    cal_end = '31/12/2017 06:00'
    obs_start = '30/01/2017 12:00'
    obs_end = '31/12/2017 06:00'
    run_id = '1'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, initialise=False)

    observations_file = os.path.join(subcatch.path_station, 'observations_synthetic.csv')
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)
    obs = obj.read_observed_streamflow(observations_file)
    print(obs)

    gzip_file(os.path.join(subcatch.path_out, run_id, 'dis.tss'))
    sim = obj.read_simulated_streamflow(run_id, cal_start, cal_end)
    print(sim)

    dates, Qsim, Qobs = obj.resample_streamflows(obs_start, obs_end, sim, obs)
    print(Qsim)
    print(Qobs)

    kge_comp = hydro_stats.fKGE(s=Qsim, o=Qobs)
    print(kge_comp)

    os.remove(os.path.join(subcatch.path_out, run_id, 'dis.tss'))

    kge_truth = [0.9613349667410458, 0.9995939501541478, 0.9949282186830977, 0.9616711994094227, 973.687533]
    print(kge_truth)
    assert np.allclose(kge_comp, kge_truth)


@pytest.mark.parametrize('dates', [('30/01/2015 06:00', '31/12/2017 06:00'), ('30/01/2017 12:00', '31/12/2020 06:00')])
def test_kge_fail(dummy_cfg, dates):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking kge function')

    obs_start = dates[0]
    obs_end = dates[1]
    print(obs_start, obs_end)

    run_id = '1'
    station_data = {}
    station_data['CAL_TYPE'] = 'NRT_6h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, initialise=False)
    subcatch.data['Obs_start'] = obs_start
    subcatch.data['Obs_end'] = obs_end
    print(subcatch.data)

    observations_file = os.path.join(subcatch.path_station, 'observations_synthetic.csv')
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)
    try:
        obs = obj.read_observed_streamflow(observations_file)
    except ValueError:
        pass
    else:
        assert False

@pytest.mark.parametrize('param', [
    (380, '0', '02/01/2009 06:00', '07/11/2017 12:00', '08/05/2012 12:00', '07/11/2017 12:00', [0.583961732802424, 0.7703130232405978, 1.3460817211190588, 1.023646876900714, 69040.253913]),
    (2733, '008133470374', '02/01/1995 06:00', '01/11/2002 00:00', '02/01/1998 06:00', '01/11/2002 00:00', [0.4332305120518942, 0.4507688484024667, 0.863698541377526, 0.9684610247282592, 7309.03849214125]),
    (892, '001497862365', '02/01/2010 06:00', '31/12/2017 00:00', '14/01/2013 00:00', '31/12/2017 00:00', [0.8087191318516062, 0.8992883303124082, 1.1618604067180205, 0.9842920769858771, 109259.8307]),
])
def test_kge_real(dummy_cfg, param):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking kge function')
    obsid = param[0]
    run_id = param[1]
    cal_start = param[2]
    cal_end = param[3]
    obs_start = param[4]
    obs_end = param[5]

    subcatch = subcatchment.SubCatchment(dummy_cfg, obsid, initialise=False)
    subcatch.data['Obs_start'] = obs_start
    subcatch.data['Obs_end'] = obs_end
    print(subcatch.data)

    dummy_cfg.observed_discharges = os.path.join(subcatch.path_station, 'observations.csv')
    gzip_file(dummy_cfg.observed_discharges)

    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)
    obs = obj.read_observed_streamflow(dummy_cfg.observed_discharges)
    print(obs)

    gzip_file(os.path.join(subcatch.path_out, run_id, 'dis.tss'))
    sim = obj.read_simulated_streamflow(run_id, cal_start, cal_end)
    print(sim)

    dates, Qsim, Qobs = obj.resample_streamflows(obs_start, obs_end, sim, obs)
    print(Qsim)
    print(Qobs)

    kge_comp = hydro_stats.fKGE(s=Qsim, o=Qobs)
    print(kge_comp)

    os.remove(dummy_cfg.observed_discharges)
    os.remove(os.path.join(subcatch.path_out, run_id, 'dis.tss'))

    kge_truth = param[6]
    print(kge_truth)
    assert np.allclose(kge_comp, kge_truth)


def test_stats(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    obsid = 380
    run_id = '0'
    obs_start = '08/05/2012 12:00'
    obs_end = '07/11/2017 12:00'
    cal_start = '02/01/2009 06:00'
    cal_end = '07/11/2017 12:00'

    subcatch = subcatchment.SubCatchment(dummy_cfg, obsid, initialise=False)
    subcatch.data['Obs_start'] = obs_start
    subcatch.data['Obs_end'] = obs_end

    dummy_cfg.observed_discharges = os.path.join(subcatch.path_station, 'observations.csv')
    gzip_file(dummy_cfg.observed_discharges)

    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)
    obj.observed_streamflow = obj.read_observed_streamflow(dummy_cfg.observed_discharges)

    gzip_file(os.path.join(subcatch.path_out, run_id, 'dis.tss'))
    sim = obj.read_simulated_streamflow(run_id, cal_start, cal_end)
    Q, stats = obj.compute_statistics(obs_start, obs_end, sim)

    assert np.isclose(stats['kge'], 0.583961732802424)
    assert np.isclose(stats['corr'], 0.7703130232405978)
    assert np.isclose(stats['bias'], 1.3460817211190588)
    assert np.isclose(stats['spread'], 1.023646876900714)
    assert np.isclose(stats['sae'], 69040.253913)
    assert np.isclose(stats['nse'], 0.195407737254919)

    os.remove(dummy_cfg.observed_discharges)
    os.remove(os.path.join(subcatch.path_out, run_id, 'dis.tss'))


def test_select_best_calib_strategy(dummy_cfg):

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data={}, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch, read_observations=False)

    # Test case where both JSDs are <= 0.1 and CORR_1st is significantly better then CORR_2nd, but KGE_bestKGE is better then KGE_1st
    JSD_1st = 0.05
    JSD_2nd = 0.04
    CORR_1st = 0.9
    CORR_2nd = 0.8
    KGE_1st = 0.7
    KGE_2nd = 0.6
    CORR_bestKGE = 0.85
    KGE_bestKGE = 0.76

    expected_result = (True, "1st", KGE_1st, CORR_1st, JSD_1st)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where both JSDs are <= 0.1 and CORR_2nd is significantly better then CORR_1st, but KGE_bestKGE is better then KGE_1st
    JSD_1st = 0.05
    JSD_2nd = 0.04
    CORR_1st = 0.8
    CORR_2nd = 0.9
    KGE_1st = 0.7
    KGE_2nd = 0.6
    CORR_bestKGE = 0.85
    KGE_bestKGE = 0.76

    expected_result = (True, "2nd", KGE_2nd, CORR_2nd, JSD_2nd)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where both JSDs are > 0.1, CORR are similare and KGE_2nd is better then KGE_1st, but CORR_bestKGE is better then CORR_2nd
    JSD_1st = 0.2
    JSD_2nd = 0.3
    CORR_1st = 0.61
    CORR_2nd = 0.60
    KGE_1st = 0.6
    KGE_2nd = 0.7
    CORR_bestKGE = 0.65
    KGE_bestKGE = 0.60

    expected_result = (True, "2nd", KGE_2nd, CORR_2nd, JSD_2nd)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where both JSDs are > 0.1, CORR are similar and KGE_1st is better then KGE_2nd, but CORR_bestKGE is better then CORR_2nd
    JSD_1st = 0.2
    JSD_2nd = 0.3
    CORR_1st = 0.61
    CORR_2nd = 0.60
    KGE_1st = 0.7
    KGE_2nd = 0.6
    CORR_bestKGE = 0.65
    KGE_bestKGE = 0.60

    expected_result = (True, "1st", KGE_1st, CORR_1st, JSD_1st)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where only JSD_1st <= 0.1 and KGE_bestKGE IS NOT significantly better then KGE_1st
    JSD_1st = 0.05
    JSD_2nd = 0.2
    CORR_1st = 0.61
    CORR_2nd = 0.60
    KGE_1st = 0.6
    KGE_2nd = 0.7
    CORR_bestKGE = 0.65
    KGE_bestKGE = 0.62

    expected_result = (False, "1st", KGE_1st, CORR_1st, JSD_1st)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where only JSD_1st <= 0.1 and KGE_bestKGE IS significantly better then KGE_1st
    JSD_1st = 0.05
    JSD_2nd = 0.2
    CORR_1st = 0.61
    CORR_2nd = 0.60
    KGE_1st = 0.6
    KGE_2nd = 0.7
    CORR_bestKGE = 0.65
    KGE_bestKGE = 0.68

    expected_result = (True, "1st", KGE_1st, CORR_1st, JSD_1st)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where only JSD_2nd <= 0.1 and KGE_bestKGE IS NOT significantly better then KGE_2nd
    JSD_1st = 0.2
    JSD_2nd = 0.05
    CORR_1st = 0.61
    CORR_2nd = 0.62
    KGE_1st = 0.6
    KGE_2nd = 0.7
    CORR_bestKGE = 0.65
    KGE_bestKGE = 0.68

    expected_result = (False, "2nd", KGE_2nd, CORR_2nd, JSD_2nd)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result

    # Test case where only JSD_2nd <= 0.1 and KGE_bestKGE IS significantly better then KGE_2nd
    JSD_1st = 0.2
    JSD_2nd = 0.05
    CORR_1st = 0.61
    CORR_2nd = 0.62
    KGE_1st = 0.6
    KGE_2nd = 0.7
    CORR_bestKGE = 0.65
    KGE_bestKGE = 0.76

    expected_result = (True, "2nd", KGE_2nd, CORR_2nd, JSD_2nd)
    assert obj.select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd, CORR_bestKGE, KGE_bestKGE) == expected_result