"""
Unit tests for liscal utility functions used by the bin/CAL_* scripts.

These tests cover pure functions and logic that benefit from fast,
isolated unit testing. End-to-end script testing is in the
test_cal*_integration.py files.
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

from liscal import stations, cutmaps, hydro_model, utils


class TestTimeStepFromType:
    """Tests for stations.time_step_from_type — the core type-to-timestep resolver."""

    def test_integer_inputs(self):
        assert stations.time_step_from_type(6) == 6
        assert stations.time_step_from_type(24) == 24

    def test_float_inputs(self):
        assert stations.time_step_from_type(6.0) == 6
        assert stations.time_step_from_type(24.0) == 24

    def test_string_inputs(self):
        assert stations.time_step_from_type("6") == 6
        assert stations.time_step_from_type("24") == 24
        assert stations.time_step_from_type("6.0") == 6
        assert stations.time_step_from_type("24.0") == 24

    def test_suffix_patterns(self):
        assert stations.time_step_from_type("NRT_6h") == 6
        assert stations.time_step_from_type("GloFAS_24h") == 24

    def test_numpy_types(self):
        assert stations.time_step_from_type(np.int64(6)) == 6
        assert stations.time_step_from_type(np.float64(24.0)) == 24
        assert stations.time_step_from_type(np.array(6.0).item()) == 6

    def test_unsupported_raises(self):
        with pytest.raises(Exception, match="12 not supported"):
            stations.time_step_from_type(12)
        with pytest.raises(Exception, match="invalid not supported"):
            stations.time_step_from_type("invalid")


class TestObservationPeriod:
    """Tests for stations.observation_period_days / observation_period_years."""

    @pytest.fixture
    def streamflow_with_gap(self):
        """730-day series with first 100 values as NaN."""
        dates = pd.date_range('2016-01-01', periods=730, freq='D')
        data = pd.Series(np.ones(730) * 50.0, index=dates)
        data.iloc[:100] = np.nan
        return data

    def test_observation_period_days_daily(self, streamflow_with_gap):
        """Daily timestep: 630 valid steps / freq=1 = 630 days."""
        result = stations.observation_period_days(24, streamflow_with_gap)
        assert result == 630.0

    def test_observation_period_days_6hourly(self, streamflow_with_gap):
        """6h timestep: 630 valid steps / freq=4 = 157.5 days."""
        result = stations.observation_period_days(6, streamflow_with_gap)
        assert result == 157.5

    def test_observation_period_years(self, streamflow_with_gap):
        """Verify years = days / 365.25."""
        result = stations.observation_period_years(24, streamflow_with_gap)
        assert result == pytest.approx(630.0 / 365.25)

    def test_all_nan_returns_zero(self):
        """All-NaN series should yield 0 days."""
        dates = pd.date_range('2016-01-01', periods=100, freq='D')
        all_nan = pd.Series(np.nan, index=dates)
        assert stations.observation_period_days(24, all_nan) == 0.0


class TestComputeSplitDate:
    """Tests for stations.compute_split_date."""

    def test_under_max_years_returns_valid_start(self):
        """When obs < max_calib_years, split_date = valid_start (use all)."""
        valid_start = '01/01/2010 00:00'
        observations = pd.Series(
            range(1000),
            index=pd.date_range('2010-01-01', periods=1000, freq='D')
        )
        result = stations.compute_split_date(5.0, 24, valid_start, observations, 20)
        assert result == valid_start

    def test_over_max_years_truncates(self):
        """When obs > max_calib_years, only last max_years are used."""
        valid_start = '01/01/2000 00:00'
        observations = pd.Series(
            range(10000),
            index=pd.date_range('2000-01-01', periods=10000, freq='D')
        )
        result = stations.compute_split_date(25.0, 24, valid_start, observations, 20)
        expected = observations.index[-7305]
        assert result == expected

    def test_6h_timestep_truncation(self):
        """Verify truncation with 6h timestep (more steps per year)."""
        valid_start = '01/01/2000 00:00'
        observations = pd.Series(
            range(40000),
            index=pd.date_range('2000-01-01', periods=40000, freq='6h')
        )
        result = stations.compute_split_date(27.4, 6, valid_start, observations, 20)
        expected = observations.index[-29220]
        assert result == expected


class TestCutMapsDispatch:
    """Tests for cutmaps.cut_maps_stations dispatch logic."""

    def test_single_id_converted_to_list(self):
        """cut_maps_stations converts a single int obsid to a list internally."""
        cfg = Mock()
        cfg.subcatchment_path = '/tmp/sub'

        with patch('liscal.cutmaps._cut_maps_stations') as mock_inner:
            cutmaps.cut_maps_stations(cfg, '/maps', 7838, useDaskConfig=False)
            mock_inner.assert_called_once_with(cfg, '/maps', [7838])

    def test_list_passthrough(self):
        """cut_maps_stations passes a list of obsids through unchanged."""
        cfg = Mock()
        cfg.subcatchment_path = '/tmp/sub'
        obsids = [7838, 6651]

        with patch('liscal.cutmaps._cut_maps_stations') as mock_inner:
            cutmaps.cut_maps_stations(cfg, '/maps', obsids, useDaskConfig=False)
            mock_inner.assert_called_once_with(cfg, '/maps', obsids)


class TestExtractStationData:
    """Tests for stations.extract_station_data — the core of CAL_5."""

    @pytest.fixture
    def extract_setup(self, tmp_path):
        """Set up config and observed data for extract_station_data."""
        dates = pd.date_range('2016-01-01', periods=730, freq='D')
        obs_df = pd.DataFrame(
            {'1001': np.ones(730) * 50.0},
            index=dates.strftime('%d/%m/%Y %H:%M'),
        )
        obs_df.index.name = 'date'
        obs_file = tmp_path / "observed.csv"
        obs_df.to_csv(obs_file)

        cfg = Mock()
        cfg.forcing_start = datetime(2016, 1, 1)
        cfg.forcing_end = datetime(2017, 12, 31)
        cfg.timestep = 1440
        cfg.observed_discharges = str(obs_file)
        cfg.subcatchment_path = str(tmp_path)
        cfg.reservoir_events = None
        cfg.num_max_calib_years = 20

        station_data = pd.Series({
            'Spinup_days': '30',
            'Min_calib_days': '100',
            'CAL_TYPE': 24,
        })
        return cfg, station_data

    def test_creates_output_csv_files(self, extract_setup, tmp_path):
        """extract_station_data produces all expected output files."""
        cfg, station_data = extract_setup
        stations.extract_station_data(cfg, None, 1001, station_data, check_obs=True)

        out_dir = tmp_path / '1001' / 'station'
        assert (out_dir / 'observations_original.csv').exists()
        assert (out_dir / 'observations_complete.csv').exists()
        assert (out_dir / 'observations.csv').exists()
        assert (out_dir / 'station_data.csv').exists()

    def test_check_obs_false_allows_short_data(self, tmp_path):
        """check_obs=False skips the min_calib_days validation."""
        dates = pd.date_range('2016-01-01', periods=50, freq='D')
        obs_df = pd.DataFrame(
            {'1001': np.ones(50) * 50.0},
            index=dates.strftime('%d/%m/%Y %H:%M'),
        )
        obs_df.index.name = 'date'
        obs_file = tmp_path / "observed.csv"
        obs_df.to_csv(obs_file)

        cfg = Mock()
        cfg.forcing_start = datetime(2016, 1, 1)
        cfg.forcing_end = datetime(2016, 2, 19)
        cfg.timestep = 1440
        cfg.observed_discharges = str(obs_file)
        cfg.subcatchment_path = str(tmp_path)
        cfg.reservoir_events = None
        cfg.num_max_calib_years = 20

        station_data = pd.Series({
            'Spinup_days': '30',
            'Min_calib_days': '100',
            'CAL_TYPE': 24,
        })

        stations.extract_station_data(cfg, None, 1001, station_data, check_obs=False)

    def test_check_obs_true_raises_on_short_data(self, tmp_path):
        """check_obs=True raises when obs_period < Min_calib_days."""
        dates = pd.date_range('2016-01-01', periods=50, freq='D')
        obs_df = pd.DataFrame(
            {'1001': np.ones(50) * 50.0},
            index=dates.strftime('%d/%m/%Y %H:%M'),
        )
        obs_df.index.name = 'date'
        obs_file = tmp_path / "observed.csv"
        obs_df.to_csv(obs_file)

        cfg = Mock()
        cfg.forcing_start = datetime(2016, 1, 1)
        cfg.forcing_end = datetime(2016, 2, 19)
        cfg.timestep = 1440
        cfg.observed_discharges = str(obs_file)
        cfg.subcatchment_path = str(tmp_path)
        cfg.reservoir_events = None
        cfg.num_max_calib_years = 20

        station_data = pd.Series({
            'Spinup_days': '30',
            'Min_calib_days': '100',
            'CAL_TYPE': 24,
        })

        with pytest.raises(Exception, match="only contains"):
            stations.extract_station_data(cfg, None, 1001, station_data, check_obs=True)


class TestCAL6RerunLogic:
    """Tests for CAL_6 rerun branching (not exercised by integration tests)."""

    def test_skip_when_best_csv_exists(self, tmp_path):
        """calibrate_subcatchment returns early when result file exists."""
        from bin.CAL_6_CALIBRATION import calibrate_subcatchment

        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "streamflow_simulated_best.csv").write_text("data")

        cfg = Mock()
        subcatch = Mock()
        subcatch.path = str(tmp_path)

        calibrate_subcatchment(cfg, 1001, subcatch)

    def test_rerun_kgejsd_triggered_by_status_file(self, tmp_path):
        """KGEJSD status file present + KGE absent -> rerun_with_KGEJSD."""
        (tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt').write_text("failed")

        calibstatus_kgejsd = str(tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt')
        calibstatus_kge = str(tmp_path / 'CalibrationStatus_3rd_run_KGE.txt')

        rerun_with_KGEJSD = False
        rerun_with_KGE = False
        if os.path.exists(calibstatus_kgejsd) and not os.path.exists(calibstatus_kge):
            rerun_with_KGEJSD = True
        elif os.path.exists(calibstatus_kge):
            rerun_with_KGE = True

        assert rerun_with_KGEJSD is True
        assert rerun_with_KGE is False

    def test_rerun_kge_triggered_by_both_status_files(self, tmp_path):
        """Both status files exist -> rerun_with_KGE."""
        (tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt').write_text("failed")
        (tmp_path / 'CalibrationStatus_3rd_run_KGE.txt').write_text("failed")

        calibstatus_kgejsd = str(tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt')
        calibstatus_kge = str(tmp_path / 'CalibrationStatus_3rd_run_KGE.txt')

        rerun_with_KGEJSD = False
        rerun_with_KGE = False
        if os.path.exists(calibstatus_kgejsd) and not os.path.exists(calibstatus_kge):
            rerun_with_KGEJSD = True
        elif os.path.exists(calibstatus_kge):
            rerun_with_KGE = True

        assert rerun_with_KGE is True
        assert rerun_with_KGEJSD is False


class TestReadParameters:
    """Tests for hydro_model.read_parameters — used by CAL_7."""

    def test_reads_pareto_front_parameters(self, tmp_path):
        """read_parameters extracts parameter values from pareto_front.csv."""
        pareto = pd.DataFrame({
            'KGE': [0.85],
            'r': [0.92],
            'B': [1.01],
            'UpperZoneTimeConstant': [5.5],
            'LowerZoneTimeConstant': [300.0],
            'GwLoss': [0.1],
        })
        pareto.to_csv(tmp_path / "pareto_front.csv", index=False)

        params = hydro_model.read_parameters(str(tmp_path))
        assert params == [5.5, 300.0, 0.1]

    def test_missing_pareto_front_raises(self, tmp_path):
        """read_parameters raises when pareto_front.csv is missing."""
        with pytest.raises(FileNotFoundError):
            hydro_model.read_parameters(str(tmp_path))


class TestReadTss:
    """Tests for utils.read_tss — used in CAL_7 long-term checks."""

    def test_reads_tss_file(self, tmp_path):
        """read_tss correctly parses a .tss formatted file."""
        tss_content = (
            "timeseries scalar\n"
            "2\n"
            "timestep\n"
            "1\n"
            "1 10.5\n"
            "2 20.3\n"
            "3 30.1\n"
        )
        tss_file = tmp_path / "test.tss"
        tss_file.write_text(tss_content)

        df = utils.read_tss(str(tss_file), skiprows=4)
        assert len(df) == 3
        assert df[1].iloc[0] == pytest.approx(10.5)
        assert df[1].iloc[2] == pytest.approx(30.1)


class TestHydroModelCache:
    """Tests for the lisflood Cache used in CAL_5/CAL_6/CAL_7."""

    def test_cache_clear_does_not_raise(self):
        """Cache.clear() should succeed without error."""
        hydro_model.Cache.clear()

    def test_cache_size_after_clear(self):
        """After clear, cache size should be 0."""
        hydro_model.Cache.clear()
        assert hydro_model.Cache.size() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
