"""
Integration tests for CAL_10_PARAMETER_MAPS.py and CAL_10_PARAMETER_MAPS_FROMCSV.py.

Calls main() for CAL_10_FROMCSV and verifies output NetCDF maps against expected values.
Tests CAL_10 set_calibrated_parameters logic (reservoir default fix).
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data')


def _create_cal10_fromcsv_test_data(tmp_path):
    """Create synthetic test data for CAL_10_PARAMETER_MAPS_FROMCSV."""
    param_ranges = pd.DataFrame({
        'MinValue': [0.01, 40, 0.5],
        'MaxValue': [40, 10000, 2],
        'DefaultValue': [10.0, 100.0, 1.0],
    }, index=pd.Index(['UpperZoneTimeConstant', 'LowerZoneTimeConstant', 'LakeMultiplier'],
                      name='ParameterName'))
    param_ranges.to_csv(tmp_path / 'param_ranges.csv')

    # Interstation NetCDF: 5x5 grid
    # 1=station_1, 2=station_2, -1=ungauged, 0=default area, NaN=outside domain
    interstation_data = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan,  1,      1,      2,     np.nan],
        [np.nan,  1,     -1,      2,     np.nan],
        [np.nan,  0,      0,      2,     np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)

    ds = xr.Dataset(
        {'Band1': (['lat', 'lon'], interstation_data)},
        coords={'lat': np.arange(5, 0, -1, dtype=np.float64),
                'lon': np.arange(1, 6, dtype=np.float64)}
    )
    interstation_path = str(tmp_path / 'interstation.nc')
    ds.to_netcdf(interstation_path)

    # Calibrated parameters (stations 1 and 2)
    calibrated = pd.DataFrame({
        'UpperZoneTimeConstant': [20.0, 30.0],
        'LowerZoneTimeConstant': [500.0, 800.0],
        'LakeMultiplier': [1.5, 0.8],
    }, index=pd.Index([1, 2], name='ID'))
    calibrated.to_csv(tmp_path / 'calibrated.csv')

    # Empty regionalisation
    regionalisation = pd.DataFrame({
        'UpperZoneTimeConstant': pd.Series(dtype=float),
        'LowerZoneTimeConstant': pd.Series(dtype=float),
        'LakeMultiplier': pd.Series(dtype=float),
    })
    regionalisation.index.name = 'ID'
    regionalisation.to_csv(tmp_path / 'regionalisation.csv')

    return interstation_path, str(tmp_path / 'param_ranges.csv'), \
           str(tmp_path / 'calibrated.csv'), str(tmp_path / 'regionalisation.csv')


class TestCAL10FromCSV:
    """Integration test calling CAL_10_PARAMETER_MAPS_FROMCSV main()."""

    def test_main_produces_correct_parameter_maps(self, tmp_path):
        """Call main() with NN interpolation and verify output values per region."""
        from CAL_10_PARAMETER_MAPS_FROMCSV import main

        interstation_path, params_path, calibrated_path, reg_path = \
            _create_cal10_fromcsv_test_data(tmp_path)

        output_dir = str(tmp_path / 'output')
        main(interstation_path, output_dir, params_path, calibrated_path, reg_path, useNN=True)

        # Verify output files
        for param in ['UpperZoneTimeConstant', 'LowerZoneTimeConstant', 'LakeMultiplier']:
            nc_file = os.path.join(output_dir, f'{param}_GloFASv5.nc')
            assert os.path.isfile(nc_file), f"main() did not produce {param}_GloFASv5.nc"

        # Load and verify UpperZoneTimeConstant
        ds = xr.open_dataset(os.path.join(output_dir, 'UpperZoneTimeConstant_GloFASv5.nc'))
        uztc = ds['UpperZoneTimeConstant'].values

        # Station 1 pixels = 20.0
        assert uztc[1, 1] == pytest.approx(20.0)
        assert uztc[1, 2] == pytest.approx(20.0)
        assert uztc[2, 1] == pytest.approx(20.0)

        # Station 2 pixels = 30.0
        assert uztc[1, 3] == pytest.approx(30.0)
        assert uztc[2, 3] == pytest.approx(30.0)
        assert uztc[3, 3] == pytest.approx(30.0)

        # Default area (ID=0) = DefaultValue = 10.0
        assert uztc[3, 1] == pytest.approx(10.0)
        assert uztc[3, 2] == pytest.approx(10.0)

        # NaN border
        assert np.isnan(uztc[0, 0])

        # -1 area with NN interpolation filled
        assert not np.isnan(uztc[2, 2])

        # Verify LakeMultiplier
        ds_lake = xr.open_dataset(os.path.join(output_dir, 'LakeMultiplier_GloFASv5.nc'))
        lake = ds_lake['LakeMultiplier'].values
        assert lake[1, 1] == pytest.approx(1.5)  # station 1
        assert lake[1, 3] == pytest.approx(0.8)  # station 2
        assert lake[3, 1] == pytest.approx(1.0)  # default

    def test_main_without_nn_leaves_gaps_as_nan(self, tmp_path):
        """Call main() without NN and verify -1 pixels are NaN."""
        from CAL_10_PARAMETER_MAPS_FROMCSV import main

        interstation_path, params_path, calibrated_path, reg_path = \
            _create_cal10_fromcsv_test_data(tmp_path)

        output_dir = str(tmp_path / 'output')
        main(interstation_path, output_dir, params_path, calibrated_path, reg_path, useNN=False)

        ds = xr.open_dataset(os.path.join(output_dir, 'UpperZoneTimeConstant_GloFASv5.nc'))
        uztc = ds['UpperZoneTimeConstant'].values

        # -1 pixel without NN = NaN
        assert np.isnan(uztc[2, 2])
        # Valid stations still correct
        assert uztc[1, 1] == pytest.approx(20.0)
        assert uztc[1, 3] == pytest.approx(30.0)


class TestCAL10PcrasterVersion:
    """Tests for CAL_10_PARAMETER_MAPS set_calibrated_parameters (reservoir default fix)."""

    def test_lakes_reservoirs_default_applies_to_current_params(self, tmp_path):
        """
        Verify lakes_reservoirs_default=True uses defaults for LakeMultiplier,
        ReservoirFloodStorage, and ReservoirFloodOutflowFactor.
        """
        import CAL_10_PARAMETER_MAPS as cal10

        class FakeScalar:
            def __init__(self, v):
                self.value = v
            def __float__(self):
                return float(self.value)

        mock_pcr = MagicMock()
        mock_pcr.scalar = lambda v: FakeScalar(float(v))
        mock_pcr.ifthenelse = lambda cond, t, f: t
        cal10.pcr = mock_pcr

        param_ranges = pd.read_csv(
            os.path.join(DATA_DIR, 'param_ranges.csv'), sep=",", index_col=0
        )

        # pareto_front.csv with all values = 999
        pareto_data = {f'param_{str(i).zfill(2)}_{param_ranges.index[i]}': [999.0]
                       for i in range(len(param_ranges))}
        pareto_data['effover(KGE)'] = [0.85]
        pareto_data['R(KGE)'] = [0.85]
        pd.DataFrame(pareto_data).to_csv(tmp_path / 'pareto_front.csv', index=True)

        params = {p: FakeScalar(0.0) for p in param_ranges.index}

        cal10.set_calibrated_parameters(
            param_ranges, 1, str(tmp_path), params, MagicMock(),
            lakes_reservoirs_default=True
        )

        # Extract values (ifthenelse returns true_val)
        captured = {p: float(params[p]) for p in param_ranges.index}

        # Reservoir/lake params use default
        assert captured['LakeMultiplier'] == pytest.approx(param_ranges.loc['LakeMultiplier', 'DefaultValue'])
        assert captured['ReservoirFloodStorage'] == pytest.approx(param_ranges.loc['ReservoirFloodStorage', 'DefaultValue'])
        assert captured['ReservoirFloodOutflowFactor'] == pytest.approx(param_ranges.loc['ReservoirFloodOutflowFactor', 'DefaultValue'])

        # Other params use pareto value (999.0)
        assert captured['UpperZoneTimeConstant'] == pytest.approx(999.0)
        assert captured['GwPercValue'] == pytest.approx(999.0)
        assert captured['b_Xinanjiang'] == pytest.approx(999.0)
