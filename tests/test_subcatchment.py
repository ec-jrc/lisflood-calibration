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

    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data={}, initialise=False)

    assert subcatch.obsid == 380
    assert subcatch.path == os.path.join(dummy_cfg.subcatchment_path, str(380))
    assert subcatch.path_out == os.path.join(dummy_cfg.subcatchment_path, str(380), 'out')


def test_subcatchment_full(dummy_cfg):

    gzip_inflow(os.path.join(dummy_cfg.subcatchment_path, str(380)))

    station_data = {
        'LisfloodX': 4307500.0,
        'LisfloodY': 2377500.0,
    }
    subcatch = subcatchment.SubCatchment(dummy_cfg, 380, station_data=station_data)

    assert subcatch.obsid == 380
    assert subcatch.path == os.path.join(dummy_cfg.subcatchment_path, str(380))
    assert subcatch.path_out == os.path.join(dummy_cfg.subcatchment_path, str(380), 'out')
    assert subcatch.path_station == os.path.join(dummy_cfg.subcatchment_path, str(380), 'station')
    assert subcatch.gaugeloc == '4307500.0 2377500.0'
    assert subcatch.inflowflag == '0'


@pytest.mark.parametrize('catch, gauge_loc', [(2823, '4187500.0 2417500.0'), (380, '4307500.0 2377500.0')])
def test_gauge_loc(dummy_cfg, catch, gauge_loc):

    station_data = {
        'LisfloodX': gauge_loc.split(' ')[0],
        'LisfloodY': gauge_loc.split(' ')[1],
    }
    subcatch = subcatchment.SubCatchment(dummy_cfg, catch, station_data=station_data, initialise=False)
    outlet_file = os.path.join(subcatch.path, "maps", "outletsmall.map")
    pcr.setclone(outlet_file)
    loc = subcatch.extract_gauge_loc(outlet_file)

    assert loc == gauge_loc


@pytest.mark.parametrize('catch, has_inflow', [(2823, 1), (380, 0)])
def test_prepare_inflows(dummy_cfg, catch, has_inflow):

    subcatch = subcatchment.SubCatchment(dummy_cfg, catch, station_data={}, initialise=False)

    inflow_dir = os.path.join(subcatch.path, 'inflow')

    os.makedirs(inflow_dir, exist_ok=True)
    inflow_flag, n_inflows = subcatch.prepare_inflows(dummy_cfg)

    assert inflow_flag == str(has_inflow)

    if inflow_flag == '1':
        chanq_truth = utils.read_tss(os.path.join(subcatch.path, 'chanq_truth.tss'), skiprows=3+n_inflows)
        chanq_check = utils.read_tss(os.path.join(inflow_dir, 'chanq.tss'), skiprows=3+n_inflows)

        assert chanq_truth.equals(chanq_check)
        os.remove(os.path.join(inflow_dir, 'chanq.tss'))
