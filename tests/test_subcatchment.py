import os
import pytest
import shutil
import gzip
import numpy as np
import pcraster as pcr

from liscal import subcatchment, utils


def gzip_inflow(subcatch_path):

    inflow_gz = os.path.join(subcatch_path, 'inflow', 'inflow.map.gz')
    inflow_map = os.path.join(subcatch_path, 'inflow', 'inflow.map')
    with gzip.open(inflow_gz, 'rb') as f_in:
        content = f_in.read()
    with open(inflow_map, 'wb') as f_out:
        f_out.write(content)

def test_subcatchment_minimal(dummy_cfg):

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, None, initialise=False)

    assert subcatch.obsid == 380
    assert subcatch.path == os.path.join(dummy_cfg.subcatchment_path, str(380))
    assert subcatch.path_out == os.path.join(dummy_cfg.subcatchment_path, str(380), 'out')


def test_subcatchment_full(dummy_cfg):

    station_data = {}
    station_data['Cal_Start'] = '29/12/1984 00:00'  # my birthday, presents accepted ^^
    station_data['Cal_End'] = '01/06/2021 00:00'

    gzip_inflow(os.path.join(dummy_cfg.subcatchment_path, str(380)))

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data)

    assert subcatch.obsid == 380
    assert subcatch.path == os.path.join(dummy_cfg.subcatchment_path, str(380))
    assert subcatch.path_out == os.path.join(dummy_cfg.subcatchment_path, str(380), 'out')
    assert subcatch.gaugeloc == '4307500.0 2377500.0'
    assert subcatch.inflowflag == '0'
    assert subcatch.cal_start == '1984-12-29 00:00'
    assert subcatch.cal_end == '2021-06-01 00:00'
    os.remove(os.path.join(subcatch.path, "inflow", "inflow_cut.map"))
    os.remove(os.path.join(subcatch.path, "inflow", "inflow.map"))


def test_calibration_start_end(dummy_cfg):

    station_data = {}
    station_data['Cal_Start'] = '29/12/1984 00:00'  # my birthday, presents accepted ^^
    station_data['Cal_End'] = '01/06/2021 00:00'

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data, initialise=False)

    cal_start, cal_end = subcatch.calibration_start_end(dummy_cfg)

    assert cal_start == '1984-12-29 00:00'
    assert cal_end == '2021-06-01 00:00'


@pytest.mark.parametrize('catch, gauge_loc', [(2823, '4187500.0 2417500.0'), (380, '4307500.0 2377500.0')])
def test_gauge_loc(dummy_cfg, catch, gauge_loc):

    subcatch = subcatchment.SubCatchment(dummy_cfg, catch, None, initialise=False)
    outlet_file = os.path.join(subcatch.path, "maps", "outletsmall.map")
    pcr.setclone(outlet_file)
    loc = subcatch.extract_gauge_loc(outlet_file)

    assert loc == gauge_loc


@pytest.mark.parametrize('catch, has_inflow', [(2823, 1), (380, 0)])
def test_prepare_inflows(dummy_cfg, catch, has_inflow):

    subcatch = subcatchment.SubCatchment(dummy_cfg, catch, None, initialise=False)

    inflow_dir = os.path.join(subcatch.path, 'inflow')

    os.makedirs(inflow_dir, exist_ok=True)
    inflow_flag, n_inflows = subcatch.prepare_inflows(dummy_cfg)

    assert inflow_flag == str(has_inflow)

    if inflow_flag == '1':
        chanq_truth = utils.read_tss(os.path.join(subcatch.path, 'chanq_truth.tss'), skiprows=3+n_inflows)
        chanq_check = utils.read_tss(os.path.join(inflow_dir, 'chanq.tss'), skiprows=3+n_inflows)

        assert chanq_truth.equals(chanq_check)
        os.remove(os.path.join(inflow_dir, 'chanq.tss'))


@pytest.mark.parametrize('catch', [2823, 380])
def test_resample_inflows(dummy_cfg, catch):
    subcatch = subcatchment.SubCatchment(dummy_cfg, catch, None, initialise=False)

    gzip_inflow(subcatch.path)

    subcatch.resample_inflows(dummy_cfg)

    pcr.setclone(os.path.join(subcatch.path, "inflow", "inflow_cut_truth.map"))
    inflow_truth = pcr.pcr2numpy(pcr.readmap(os.path.join(subcatch.path, "inflow", "inflow_cut_truth.map")), mv=-1)
    inflow_check = pcr.pcr2numpy(pcr.readmap(os.path.join(subcatch.path, "inflow", "inflow_cut.map")), mv=-1)

    assert np.array_equal(inflow_truth, inflow_check)
    os.remove(os.path.join(subcatch.path, "inflow", "inflow_cut.map"))
    os.remove(os.path.join(subcatch.path, "inflow", "inflow.map"))


def test_resample_inflows_missing(dummy_cfg):
    subcatch = subcatchment.SubCatchment(dummy_cfg, 2824, None, initialise=False)
    
    # first clean up inflow in case directory is not empty
    inflow_dir = os.path.join(subcatch.path, 'inflow')
    shutil.rmtree(inflow_dir, ignore_errors=True)
    os.makedirs(inflow_dir, exist_ok=True)

    # first try when both inflow and masks missing
    try:
        subcatch.resample_inflows(dummy_cfg)
    except FileNotFoundError as error:
        check = str(error)
        assert check.startswith('inflow map missing')

    # now check with a dummy inflow file to get the mask error message
    open(os.path.join(inflow_dir, "inflow.map"), 'a').close()
    try:
        subcatch.resample_inflows(dummy_cfg)
    except FileNotFoundError as error:
        check = str(error)
        assert check.startswith('mask map missing')
    shutil.rmtree(inflow_dir)
