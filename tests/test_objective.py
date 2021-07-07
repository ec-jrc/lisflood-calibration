import os
import gzip
from datetime import datetime
import numpy as np
import xarray as xr
from liscal import subcatchment, objective, utils, hydro_stats
import matplotlib.pyplot as plt


def test_phistory_ranked(dummy_cfg):

    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    print('checking pHistory file')

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, None, initialise=False)
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

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, None, initialise=False)
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


# def test_kge(dummy_cfg):
#     path_subcatch = dummy_cfg.path_subcatch
#     param_ranges = dummy_cfg.param_ranges
#     path_out = dummy_cfg.path_out

#     dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'Qtss_380.csv')
#     dummy_cfg.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")
#     dummy_cfg.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
#     # dummy_cfg.forcing_start = datetime.strptime('02/01/2009 06:00', "%d/%m/%Y %H:%M")
#     # dummy_cfg.forcing_end = datetime.strptime('02/01/2015 06:00', "%d/%m/%Y %H:%M")
#     dummy_cfg.spinup_days = 1095

#     print('checking kge function')

#     station_data = {}
#     station_data['Cal_Start'] = '02/01/2009 06:00'
#     station_data['Cal_End'] = '07/11/2017 12:00'
#     station_data['CAL_TYPE'] = 'NRT_6h'

#     subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data, initialise=False)
#     obj = objective.ObjectiveKGE(dummy_cfg, subcatch)
#     print(obj.observed_streamflow)
#     obs = obj.observed_streamflow[~np.isnan(obj.observed_streamflow)]
#     print(obs)

#     sim = obj.read_simulated_streamflow('0')
#     print(sim)
#     sim.index = obj.observed_streamflow.index
#     print(sim[~np.isnan(obj.observed_streamflow)])

#     Qsim, Qobs = obj.resample_streamflows(sim, obj.observed_streamflow)
#     print(Qsim)
#     print(Qobs)

#     kge_comp = obj.compute_KGE(Qsim, Qobs)
#     print(kge_comp)

#     assert False


def test_kge_synthetic(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'convergenceTester.csv')
    gzip_file(dummy_cfg.observed_discharges)

    print('checking kge function')

    station_data = {}
    station_data['Cal_Start'] = '31/12/2016 06:00'
    station_data['Cal_End'] = '31/12/2017 06:00'
    station_data['CAL_TYPE'] = 'NRT_6h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch)
    print(obj.observed_streamflow)

    gzip_file(os.path.join(subcatch.path_out, 'dis1.tss'))
    sim = obj.read_simulated_streamflow('1')
    print(sim)

    Qsim, Qobs = obj.resample_streamflows(sim, obj.observed_streamflow)
    print(Qsim)
    print(Qobs)

    kge_comp = hydro_stats.fKGE(s=Qsim, o=Qobs, spinup=subcatch.spinup)
    print(kge_comp)

    os.remove(dummy_cfg.observed_discharges)
    os.remove(os.path.join(subcatch.path_out, 'dis1.tss'))

    assert kge_comp == (0.9613349667410458, 0.9995939501541478, 0.9949282186830977, 0.9616711994094227, 973.489873)


def test_kge_fail(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'Qtss_380_fail.csv')
    gzip_file(dummy_cfg.observed_discharges)
    dummy_cfg.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.spinup_days = 1095

    print('checking if wrong dates in observations generates a ValueError')

    station_data = {}
    station_data['Cal_Start'] = '02/01/2009 06:00'
    station_data['Cal_End'] = '07/11/2017 12:00'
    station_data['CAL_TYPE'] = 'NRT_6h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data, initialise=False)
    try:
        obj = objective.ObjectiveKGE(dummy_cfg, subcatch)
    except ValueError:
        pass
    else:
        assert False

    os.remove(dummy_cfg.observed_discharges)


def test_kge_real(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'Qtss_380.csv')
    gzip_file(dummy_cfg.observed_discharges)
    dummy_cfg.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.spinup_days = 1095

    print('checking kge function')

    station_data = {}
    station_data['Cal_Start'] = '02/01/2009 06:00'
    station_data['Cal_End'] = '07/11/2017 12:00'
    station_data['CAL_TYPE'] = 'NRT_6h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch)
    print(obj.observed_streamflow)

    gzip_file(os.path.join(subcatch.path_out, 'dis0.tss'))
    sim = obj.read_simulated_streamflow('0')
    print(sim)

    Qsim, Qobs = obj.resample_streamflows(sim, obj.observed_streamflow)
    print(Qsim)
    print(Qobs)

    kge_comp = hydro_stats.fKGE(s=Qsim, o=Qobs, spinup=subcatch.spinup)
    print(kge_comp)

    os.remove(dummy_cfg.observed_discharges)
    os.remove(os.path.join(subcatch.path_out, 'dis0.tss'))

    assert kge_comp == (0.583961732802424, 0.7703130232405978, 1.3460817211190588, 1.023646876900714, 69039.41991299999)


def test_kge_24h(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    dummy_cfg.path_subcatch = os.path.join(dummy_cfg.subcatchment_path, '2733')
    dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'Qtss_2733.csv')
    gzip_file(dummy_cfg.observed_discharges)
    dummy_cfg.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.spinup_days = 1095

    print('checking kge function')

    station_data = {}
    station_data['Cal_Start'] = '02/01/1995 06:00'
    station_data['Cal_End'] = '01/11/2002 00:00'
    station_data['CAL_TYPE'] = 'HIST_24h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 2733, station_data, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch)
    print(obj.observed_streamflow)

    gzip_file(os.path.join(subcatch.path_out, 'dis008133470374.tss'))
    sim = obj.read_simulated_streamflow('008133470374')
    print(sim)

    Qsim, Qobs = obj.resample_streamflows(sim, obj.observed_streamflow)
    print(Qsim[dummy_cfg.spinup_days:])
    print(Qobs[dummy_cfg.spinup_days:])

    kge_comp = hydro_stats.fKGE(s=Qsim, o=Qobs, spinup=subcatch.spinup)
    print(kge_comp)

    os.remove(dummy_cfg.observed_discharges)
    os.remove(os.path.join(subcatch.path_out, 'dis008133470374.tss'))

    assert kge_comp == (0.4332305120518942, 0.4507688484024667, 0.863698541377526, 0.9684610247282592, 7304.1193571412505)


def test_kge_6h(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    dummy_cfg.path_subcatch = os.path.join(dummy_cfg.subcatchment_path, '892')
    dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'Qtss_892.csv')
    gzip_file(dummy_cfg.observed_discharges)
    dummy_cfg.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.spinup_days = 1095

    print('checking kge function')

    station_data = {}
    station_data['Cal_Start'] = '02/01/2010 06:00'
    station_data['Cal_End'] = '31/12/2017 00:00'
    station_data['CAL_TYPE'] = 'NRT_6h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 892, station_data, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch)
    print(obj.observed_streamflow)

    gzip_file(os.path.join(subcatch.path_out, 'dis001497862365.tss'))
    sim = obj.read_simulated_streamflow('001497862365')
    print(sim)

    Qsim, Qobs = obj.resample_streamflows(sim, obj.observed_streamflow)
    print(Qsim[dummy_cfg.spinup_days:])
    print(Qobs[dummy_cfg.spinup_days:])

    kge_comp = hydro_stats.fKGE(s=Qsim, o=Qobs, spinup=subcatch.spinup)
    print(kge_comp)

    os.remove(dummy_cfg.observed_discharges)
    os.remove(os.path.join(subcatch.path_out, 'dis001497862365.tss'))

    assert kge_comp == (0.8087191318516062, 0.8992883303124082, 1.1618604067180205, 0.9842920769858771, 109250.76130000001)


def test_stats(dummy_cfg):
    path_subcatch = dummy_cfg.path_subcatch
    param_ranges = dummy_cfg.param_ranges
    path_out = dummy_cfg.path_out

    dummy_cfg.observed_discharges = os.path.join(dummy_cfg.path_subcatch, 'Qtss_380.csv')
    gzip_file(dummy_cfg.observed_discharges)
    dummy_cfg.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
    dummy_cfg.spinup_days = 1095

    print('checking kge function')

    station_data = {}
    station_data['Cal_Start'] = '02/01/2009 06:00'
    station_data['Cal_End'] = '07/11/2017 12:00'
    station_data['CAL_TYPE'] = 'NRT_6h'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data, initialise=False)
    obj = objective.ObjectiveKGE(dummy_cfg, subcatch)

    gzip_file(os.path.join(subcatch.path_out, 'dis0.tss'))
    Q, stats = obj.compute_statistics('0')

    assert stats['kge'] == 0.583961732802424
    assert stats['corr'] == 0.7703130232405978
    assert stats['bias'] == 1.3460817211190588
    assert stats['spread'] == 1.023646876900714
    assert stats['sae'] == 69039.41991299999
    assert stats['nse'] == 0.195407737254919

    os.remove(os.path.join(subcatch.path_out, 'dis0.tss'))
