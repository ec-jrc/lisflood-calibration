"""
Unit tests for hydro_model.filter_param_ranges — edge cases that cannot be
tested with the real LISFLOOD model on catchment 7838.

The main drop/keep scenarios are covered by integration tests in
test_cal6_integration.py::TestCAL6ParamFiltering and TestCAL6ReservoirParamFiltering.

These tests cover only:
- Lake split into per-lake parameters (no multi-lake catchment in test data)
- Snow threshold boundary (min_TAvgS == TempSnow → kept, not dropped)
- TransSub threshold boundary (min_AridIdx == 0.5 → dropped, >= semantics)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from liscal.hydro_model import filter_param_ranges


@pytest.fixture
def param_ranges_all():
    """A param_ranges DataFrame with all droppable parameters."""
    data = {
        'MinValue':     [0.01, 40,   0.01, 0,    0.01, 0.5, 2.5, 0.5, 0.5,  0.2,  0.1, 0.5, 0,   0.15],
        'MaxValue':     [40,   730,  2,    30,   5,    8,   6.5, 5,   2,    0.99, 0.5, 5,   1,   0.15],
        'DefaultValue': [10,   100,  0.8,  10,   0.5,  4,   4,   1,   1,    0.75, 0.3, 1,   0,   0],
    }
    index = [
        'UpperZoneTimeConstant', 'LowerZoneTimeConstant', 'GwPercValue',
        'LZThreshold', 'b_Xinanjiang', 'PowerPrefFlow', 'SnowMeltCoef',
        'CalChanMan1', 'LakeMultiplier', 'ReservoirFloodStorage',
        'ReservoirFloodOutflowFactor', 'CalChanMan3', 'GwLoss', 'TransSub',
    ]
    return pd.DataFrame(data, index=index)


def _make_cfg(param_ranges, use_aridity_index_check=False):
    cfg = Mock()
    cfg.param_ranges = param_ranges.copy()
    cfg.use_aridity_index_check = use_aridity_index_check
    return cfg


def _make_model(options, station_data_path, binding=None):
    model = Mock()
    model.lissettings = Mock()
    model.lissettings.options = options
    model.lissettings.binding = binding or {'TempSnow': '1.0'}
    model.subcatch = Mock()
    model.subcatch.path_station = station_data_path
    return model


def _write_station_data(tmp_path, min_tavgs='-5.0', min_arid_idx='0.8'):
    station_csv = tmp_path / "station_data.csv"
    pd.DataFrame(
        {'value': [min_tavgs, min_arid_idx]},
        index=['min_TAvgS', 'min_AridIdx']
    ).to_csv(station_csv)


class TestLakeSplitPerLake:
    """Lake split into per-lake parameters — no multi-lake catchment in test data."""

    def test_lake_split_creates_per_lake_params(self, param_ranges_all, tmp_path):
        """split_lake_params=True replaces LakeMultiplier with per-lake entries."""
        _write_station_data(tmp_path)

        cfg = _make_cfg(param_ranges_all)
        model = _make_model(
            options={'simulateLakes': True, 'simulateReservoirs': True, 'MCTRouting': True},
            station_data_path=str(tmp_path),
        )

        # Simulate 3 lake sites (IDs: 101, 102, 103) on channels
        lake_sites = np.array([0, 101, 0, 102, 103, 0], dtype=float)
        channels_bool = np.array([False, True, True, True, True, False])

        with patch('lisflood.global_modules.add1.loadmap', return_value=lake_sites.copy()), \
             patch('lisflood.global_modules.add1.compressArray', return_value=channels_bool), \
             patch('pcraster.boolean'):
            filter_param_ranges(cfg, model, split_lake_params=True)

        assert 'LakeMultiplier' not in cfg.param_ranges.index
        assert 'LakeMultiplier_101' in cfg.param_ranges.index
        assert 'LakeMultiplier_102' in cfg.param_ranges.index
        assert 'LakeMultiplier_103' in cfg.param_ranges.index


class TestSnowThresholdBoundary:
    """Boundary: min_TAvgS == TempSnow means SnowMeltCoef is kept (strict >)."""

    def test_snow_param_kept_at_exact_threshold(self, param_ranges_all, tmp_path):
        """SnowMeltCoef stays when min_TAvgS == TempSnow (condition is >, not >=)."""
        _write_station_data(tmp_path, min_tavgs='1.0')

        cfg = _make_cfg(param_ranges_all)
        model = _make_model(
            options={'simulateLakes': True, 'simulateReservoirs': True, 'MCTRouting': True},
            station_data_path=str(tmp_path),
            binding={'TempSnow': '1.0'},
        )

        filter_param_ranges(cfg, model, split_lake_params=False)

        assert 'SnowMeltCoef' in cfg.param_ranges.index


class TestTransSubThresholdBoundary:
    """Boundary: min_AridIdx == 0.5 means TransSub is dropped (condition is >=)."""

    def test_transsub_dropped_at_exact_threshold(self, param_ranges_all, tmp_path):
        """TransSub dropped when min_AridIdx == 0.5 (>= semantics)."""
        _write_station_data(tmp_path, min_arid_idx='0.5')

        cfg = _make_cfg(param_ranges_all, use_aridity_index_check=True)
        model = _make_model(
            options={'simulateLakes': True, 'simulateReservoirs': True, 'MCTRouting': True},
            station_data_path=str(tmp_path),
        )

        filter_param_ranges(cfg, model, split_lake_params=False)

        assert 'TransSub' not in cfg.param_ranges.index
