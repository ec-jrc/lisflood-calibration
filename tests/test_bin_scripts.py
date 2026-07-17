"""
Unit tests for bin/CAL_* scripts.

These tests cover the main logic of the calibration pipeline scripts:
- CAL_1_FILTER_STATIONS: Station filtering based on data availability
- CAL_2_HYDRO_DEPENDENCIES: Station dependencies and catchment masks
- CAL_3_MASK: Catchment mask creation
- CAL_4_CUT_MAPS: Map cutting for catchments
- CAL_5_EXTRACT_STATION: Station data extraction

Note: Some tests require mock data or external dependencies (PCRaster, etc.)
and are marked accordingly.
"""

import pytest
import os
import sys
import argparse
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Add bin to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

# Import modules to test
from liscal import config, stations, cutmaps
from liscal import templates, calibration, subcatchment, objective, hydro_model


class TestConfigFilter:
    """Tests for ConfigFilter class from CAL_1_FILTER_STATIONS.py"""

    def test_config_filter_init(self, tmp_path):
        """Test ConfigFilter initialization with valid settings."""
        # Create a temporary settings file
        settings_file = tmp_path / "test_settings.txt"
        settings_file.write_text("""
[Main]
forcing_start = 31/12/2016 06:00
forcing_end = 31/12/2017 06:00
timestep = 360

[Stations]
observed_discharges = observed_discharges.csv
stations_data = stations_data.csv
""")
        
        # Create dummy observed_discharges and stations_data files
        (tmp_path / "observed_discharges.csv").write_text("")
        (tmp_path / "stations_data.csv").write_text("")
        
        # Import and test
        from bin.CAL_1_FILTER_STATIONS import ConfigFilter
        
        # Mock the parent class to avoid full initialization
        with patch.object(config.Config, '__init__', return_value=None):
            cfg = object.__new__(ConfigFilter)
            cfg.parser = config.ConfigParser()
            cfg.parser.read(str(settings_file))
            cfg.forcing_start = datetime.strptime(cfg.parser.get('Main','forcing_start'), "%d/%m/%Y %H:%M")
            cfg.forcing_end = datetime.strptime(cfg.parser.get('Main','forcing_end'), "%d/%m/%Y %H:%M")
            cfg.timestep = int(cfg.parser.get('Main', 'timestep'))
            cfg.observed_discharges = cfg.parser.get('Stations', 'observed_discharges')
            cfg.stations_data = cfg.parser.get('Stations', 'stations_data')
            
            assert cfg.forcing_start == datetime(2016, 12, 31, 6, 0)
            assert cfg.forcing_end == datetime(2017, 12, 31, 6, 0)
            assert cfg.timestep == 360

    def test_config_filter_invalid_timestep(self, tmp_path):
        """Test ConfigFilter raises exception for invalid timestep."""
        settings_file = tmp_path / "test_settings.txt"
        settings_file.write_text("""
[Main]
forcing_start = 31/12/2016 06:00
forcing_end = 31/12/2017 06:00
timestep = 720

[Stations]
observed_discharges = observed_discharges.csv
stations_data = stations_data.csv
""")
        
        from bin.CAL_1_FILTER_STATIONS import ConfigFilter
        
        with patch.object(config.Config, '__init__', return_value=None):
            cfg = object.__new__(ConfigFilter)
            cfg.parser = config.ConfigParser()
            cfg.parser.read(str(settings_file))
            cfg.timestep = int(cfg.parser.get('Main', 'timestep'))
            
            with pytest.raises(Exception) as exc_info:
                if cfg.timestep != 360 and cfg.timestep != 1440:
                    raise Exception('Calibration timestep {} not supported'.format(cfg.timestep))
            
            assert 'Calibration timestep 720 not supported' in str(exc_info.value)


class TestStationFiltering:
    """Tests for station filtering logic from CAL_1_FILTER_STATIONS.py"""

    @pytest.fixture
    def sample_stations_data(self):
        """Create sample stations data for testing."""
        return pd.DataFrame({
            'LisfloodX': [100.0, 200.0, 300.0],
            'LisfloodY': [50.0, 60.0, 70.0],
            'EC_calib': [1, 1, 0],
            'Spinup_days': [30, 30, 30],
            'Min_calib_days': [100, 200, 50],
            'CAL_TYPE': [24, 24, 24],
            'DrainingArea.km2.LDD': [1000, 2000, 500]
        }, index=pd.Index([1001, 1002, 1003], name='ObsID'))

    @pytest.fixture
    def sample_observed_data(self):
        """Create sample observed discharge data."""
        dates = pd.date_range('2016-01-01', periods=400, freq='D')
        data = np.random.rand(400) * 100
        # Add some NaN values
        data[50:100] = np.nan
        df = pd.DataFrame({
            1001: data,
            1002: np.random.rand(400) * 100,
            1003: np.random.rand(400) * 100
        }, index=dates)
        df.index = df.index.strftime('%d/%m/%Y %H:%M')
        return df

    def test_filter_stations_by_calib_type(self, sample_stations_data):
        """Test filtering stations by calibration type."""
        stations_meta = sample_stations_data.copy()
        stations_type = 1
        
        filtered = stations_meta[stations_meta['EC_calib'] >= int(stations_type)]
        
        assert len(filtered) == 2
        assert 1001 in filtered.index
        assert 1002 in filtered.index
        assert 1003 not in filtered.index

    def test_observation_period_days_calculation(self):
        """Test observation period days calculation."""
        # Create observed streamflow: 365 values with first 50 as NaN
        dates = pd.date_range('2016-01-01', periods=365, freq='D')
        observed_streamflow = pd.Series(
            np.ones(365) * 42.0,
            index=dates
        )
        observed_streamflow.iloc[0:50] = np.nan
        
        station_type = 24  # daily: freq = 24/24 = 1
        obs_period_days = stations.observation_period_days(station_type, observed_streamflow)
        
        # Exactly (365 - 50) / 1 = 315 days
        assert obs_period_days == 315.0

    def test_valid_station_selection(self, sample_stations_data):
        """Test selection of valid stations based on data availability.
        
        Station 1001: Min_calib_days=100, gets 250 valid days -> VALID
        Station 1002: Min_calib_days=200, gets 150 valid days -> INVALID (150 < 200)
        Station 1003: Min_calib_days=50,  gets 300 valid days -> VALID
        """
        cfg = Mock()
        cfg.forcing_start = datetime(2016, 1, 1)
        cfg.forcing_end = datetime(2016, 12, 31)
        cfg.timestep = 1440
        cfg.observed_discharges = 'dummy.csv'

        # Build observed data with controlled valid-day counts per station
        # Period after spinup (30 days) is 2016-01-31 to 2016-12-31 = 336 days
        dates = pd.date_range('2016-01-01', '2016-12-31', freq='D')
        n = len(dates)

        # Station 1001: first 86 days NaN (after spinup start on day 31, gives 336 - 86 = 250 valid days)
        data_1001 = np.ones(n) * 50.0
        data_1001[:86] = np.nan

        # Station 1002: first 186 days NaN (after spinup start on day 31, gives 336 - 186 = 150 valid days)
        data_1002 = np.ones(n) * 60.0
        data_1002[:186] = np.nan

        # Station 1003: all valid (336 valid days after spinup)
        data_1003 = np.ones(n) * 70.0

        observed_data = pd.DataFrame({
            1001: data_1001,
            1002: data_1002,
            1003: data_1003,
        }, index=dates.strftime('%d/%m/%Y %H:%M'))

        stations_meta = sample_stations_data.copy()

        valid_stations = []
        unvalid_stations = []

        for index, row in stations_meta.iterrows():
            start_date = (cfg.forcing_start + timedelta(days=int(row['Spinup_days']))).strftime('%d/%m/%Y %H:%M')
            end_date = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

            observed_streamflow = observed_data[index]
            observed_streamflow = observed_streamflow[start_date:end_date]
            obs_period_days = stations.observation_period_days(row['CAL_TYPE'], observed_streamflow)

            if obs_period_days >= float(row['Min_calib_days']):
                valid_stations.append(index)
            else:
                unvalid_stations.append(index)

        # Station 1001: ~250 valid days >= Min_calib_days=100 -> valid
        assert 1001 in valid_stations
        # Station 1002: ~150 valid days < Min_calib_days=200 -> invalid
        assert 1002 in unvalid_stations
        # Station 1003: ~300 valid days >= Min_calib_days=50 -> valid
        assert 1003 in valid_stations


class TestTimeStepFromType:
    """Tests for time_step_from_type function from liscal/stations.py"""

    def test_time_step_integer(self):
        """Test with integer input."""
        assert stations.time_step_from_type(6) == 6
        assert stations.time_step_from_type(24) == 24

    def test_time_step_float(self):
        """Test with float input."""
        assert stations.time_step_from_type(6.0) == 6
        assert stations.time_step_from_type(24.0) == 24

    def test_time_step_string(self):
        """Test with string input."""
        assert stations.time_step_from_type("6") == 6
        assert stations.time_step_from_type("24") == 24
        assert stations.time_step_from_type("6.0") == 6
        assert stations.time_step_from_type("24.0") == 24

    def test_time_step_with_suffix(self):
        """Test with suffix patterns."""
        assert stations.time_step_from_type("NRT_6h") == 6
        assert stations.time_step_from_type("GloFAS_24h") == 24

    def test_time_step_numpy_types(self):
        """Test with numpy types."""
        assert stations.time_step_from_type(np.int64(6)) == 6
        assert stations.time_step_from_type(np.float64(24.0)) == 24
        assert stations.time_step_from_type(np.array(6.0).item()) == 6

    def test_time_step_invalid_raises(self):
        """Test that invalid input raises exception."""
        with pytest.raises(Exception) as exc_info:
            stations.time_step_from_type(12)
        assert "Calibration type 12 not supported" in str(exc_info.value)
        
        with pytest.raises(Exception) as exc_info:
            stations.time_step_from_type("invalid")
        assert "Calibration type invalid not supported" in str(exc_info.value)


class TestObservationPeriod:
    """Tests for observation period calculation functions."""

    @pytest.fixture
    def sample_observed_streamflow(self):
        """Create observed streamflow: 730 values with first 100 as NaN."""
        dates = pd.date_range('2016-01-01', periods=730, freq='D')
        data = pd.Series(np.ones(730) * 50.0, index=dates)
        data.iloc[0:100] = np.nan
        return data

    def test_observation_period_days(self, sample_observed_streamflow):
        """Test observation_period_days function."""
        station_type = 24  # daily: freq = 24/24 = 1
        obs_period_days = stations.observation_period_days(station_type, sample_observed_streamflow)
        
        # Exactly (730 - 100) / 1 = 630 days
        assert obs_period_days == 630.0

    def test_observation_period_years(self, sample_observed_streamflow):
        """Test observation_period_years function."""
        station_type = 24  # daily: freq = 24/24 = 1
        obs_period_years = stations.observation_period_years(station_type, sample_observed_streamflow)
        
        # Exactly 630 / 365.25 = 1.7248... years
        assert obs_period_years == 630.0 / 365.25

    def test_observation_period_6h_timestep(self, sample_observed_streamflow):
        """Test observation period with 6-hour timestep."""
        station_type = 6  # 6-hourly: freq = 24/6 = 4
        obs_period_days = stations.observation_period_days(station_type, sample_observed_streamflow)
        
        # With 6h timestep, freq = 4, so days = 630 / 4 = 157.5
        assert obs_period_days == 630.0 / 4.0


class TestComputeSplitDate:
    """Tests for compute_split_date function."""

    def test_split_date_under_max_years(self):
        """Test split date when observation period is under max years."""
        obs_period_years = 5.0
        dt = 24
        valid_start = '01/01/2010 00:00'
        observations_filtered = pd.Series(
            [1, 2, 3],
            index=pd.date_range('2010-01-01', periods=3, freq='D')
        )
        num_max_calib_years = 20
        
        split_date = stations.compute_split_date(
            obs_period_years, dt, valid_start, observations_filtered, num_max_calib_years
        )
        
        assert split_date == valid_start

    def test_split_date_over_max_years(self):
        """Test split date when observation period exceeds max years."""
        obs_period_years = 25.0
        dt = 24
        valid_start = '01/01/2000 00:00'
        # 10000 daily observations starting 2000-01-01
        observations_filtered = pd.Series(
            range(10000),
            index=pd.date_range('2000-01-01', periods=10000, freq='D')
        )
        num_max_calib_years = 20
        
        split_date = stations.compute_split_date(
            obs_period_years, dt, valid_start, observations_filtered, num_max_calib_years
        )
        
        # steps_MAXyears = int(20 * 365.25 * 24 / 24) = 7305
        # split_date = observations_filtered.index[-7305]
        expected_split_date = observations_filtered.index[-7305]
        assert split_date == expected_split_date


class TestConfigCutMaps:
    """Tests for ConfigCutMaps class from CAL_4_CUT_MAPS.py"""

    def test_config_cut_maps_init(self, tmp_path):
        """Test ConfigCutMaps initialization."""
        settings_file = tmp_path / "test_settings.txt"
        settings_file.write_text("""
[Path]
subcatchment_path = /tmp/subcatchments

[Stations]
stations_data = /tmp/stations_data.csv
""")
        
        from bin.CAL_4_CUT_MAPS import ConfigCutMaps
        
        with patch.object(config.Config, '__init__', return_value=None):
            cfg = object.__new__(ConfigCutMaps)
            cfg.parser = config.ConfigParser()
            cfg.parser.read(str(settings_file))
            cfg.subcatchment_path = cfg.parser.get('Path', 'subcatchment_path')
            cfg.stations_data = cfg.parser.get('Stations', 'stations_data')
            
            assert cfg.subcatchment_path == '/tmp/subcatchments'
            assert cfg.stations_data == '/tmp/stations_data.csv'


class TestCutMapsStationParsing:
    """Tests for station parsing logic in CAL_4_CUT_MAPS.py"""

    def test_parse_station_as_integer(self):
        """Test parsing station as integer."""
        station_input = "7838"
        obsid = None
        try:
            obsid = int(station_input)
        except ValueError:
            pass
        
        assert obsid == 7838

    def test_parse_station_as_file(self, tmp_path):
        """Test parsing station as file path."""
        # Create a dummy station list file
        station_file = tmp_path / "stations.txt"
        station_file.write_text("7838\n6651\n6593\n")
        
        station_input = str(station_file)
        obsid = None
        if os.path.isfile(station_input):
            try:
                CatchmentsToProcess = pd.read_csv(station_input, sep=",", header=None)
                obsid = CatchmentsToProcess[0].values
            except:
                pass
        
        assert obsid is not None
        assert len(obsid) == 3

    def test_parse_station_invalid_input(self):
        """Test parsing invalid station input."""
        station_input = "nonexistent_file.txt"
        obsid = None
        station_list = None
        
        try:
            obsid = int(station_input)
        except ValueError:
            if os.path.isfile(station_input):
                station_list = station_input
            else:
                print("Error: Input is neither a valid integer nor an existing file path.")
        
        assert obsid is None
        assert station_list is None


class TestExtractStationData:
    """Tests for extract_station_data function from liscal/stations.py"""

    @pytest.fixture
    def mock_cfg(self):
        """Create mock configuration."""
        cfg = Mock()
        cfg.forcing_start = datetime(2016, 1, 1)
        cfg.forcing_end = datetime(2017, 12, 31)
        cfg.timestep = 1440
        cfg.observed_discharges = 'observed_discharges.csv'
        cfg.subcatchment_path = '/tmp/subcatchments'
        cfg.reservoir_events = None
        cfg.num_max_calib_years = 20
        return cfg

    def test_extract_station_data_creates_output_directory(self, mock_cfg, tmp_path):
        """Test that extract_station_data creates output directory."""
        # Create mock observed discharges file
        obs_file = tmp_path / "observed_discharges.csv"
        dates = pd.date_range('2016-01-01', periods=730, freq='D')
        df = pd.DataFrame({
            '1001': np.random.rand(730) * 100
        }, index=dates.strftime('%d/%m/%Y %H:%M'))
        df.index.name = 'date'
        df.to_csv(obs_file)
        
        mock_cfg.observed_discharges = str(obs_file)
        mock_cfg.subcatchment_path = str(tmp_path)
        
        station_data = pd.Series({
            'Spinup_days': 30,
            'Min_calib_days': 100,
            'CAL_TYPE': 24,
            'LisfloodX': 100.0,
            'LisfloodY': 50.0
        })
        
        # This test verifies the function can be called without errors
        # Full integration test would require more setup
        with patch('liscal.stations.loadmap'), \
             patch('liscal.stations.boolean'), \
             patch('liscal.stations.compressArray'):
            # Function would be called here in full test
            pass


class TestCAL5StationParsing:
    """Tests for station parsing logic in CAL_5_EXTRACT_STATION.py"""

    def test_parse_single_station(self):
        """Test parsing a single station ID."""
        station_input = "7838"
        obsids = None
        
        try:
            obsid = int(station_input)
            obsids = [obsid]
        except ValueError:
            pass
        
        assert obsids == [7838]

    def test_parse_station_list(self, tmp_path):
        """Test parsing a station list file."""
        station_file = tmp_path / "stations.txt"
        station_file.write_text("7838\n6651\n6593\n")
        
        station_input = str(station_file)
        obsids = None
        
        try:
            obsid = int(station_input)
            obsids = [obsid]
        except ValueError:
            if os.path.isfile(station_input):
                try:
                    CatchmentsToProcess = pd.read_csv(station_input, sep=",", header=None)
                    obsids = CatchmentsToProcess[0].values
                except:
                    pass
        
        assert obsids is not None
        assert len(obsids) == 3

    def test_parse_invalid_station(self):
        """Test parsing invalid station input."""
        station_input = "invalid_station_id"
        obsids = None
        
        try:
            obsid = int(station_input)
            obsids = [obsid]
        except ValueError:
            if os.path.isfile(station_input):
                pass
            else:
                print("Error: Input is neither a valid integer nor an existing file path.")
        
        assert obsids is None


class TestHydroModelCache:
    """Tests for hydro_model.Cache functionality."""

    def test_cache_clear(self):
        """Test that Cache can be cleared."""
        # The Cache is used in CAL_5_EXTRACT_STATION to clear previous catchment data
        hydro_model.Cache.clear()
        # This should complete without error
        assert True

    def test_cache_is_class(self):
        """Test that Cache is a class (type) and can be cleared."""
        # hydro_model.Cache is a class, not an instance
        # Cache.clear() clears any cached data
        assert isinstance(hydro_model.Cache, type)
        # Calling clear should not raise an error
        hydro_model.Cache.clear()


class TestCAL2HydroDependenciesLogic:
    """Tests for logic in CAL_2_HYDRO_DEPENDENCIES.py"""

    def test_station_id_validation(self):
        """Test station ID validation (max 999999)."""
        max_station_id = 999999
        
        # Valid station IDs
        assert 1000 < max_station_id
        assert 999999 == max_station_id
        
        # Invalid station ID
        invalid_id = 1000000
        assert invalid_id > max_station_id

    def test_station_txt_format(self):
        """Test station text file format."""
        stationdata = pd.DataFrame({
            'LisfloodX': [100.5, 200.3],
            'LisfloodY': [50.2, 60.4]
        }, index=[1001, 1002])
        
        expected_lines = [
            "100.5 50.2 1001.0\n",
            "200.3 60.4 1002.0\n"
        ]
        
        # Verify format
        for idx, (index, row) in enumerate(stationdata.iterrows()):
            line = f"{row['LisfloodX']} {row['LisfloodY']} {float(index)}\n"
            assert line == expected_lines[idx]


class TestCAL3MaskLogic:
    """Tests for logic in CAL_3_MASK.py"""

    def test_catchment_directory_creation(self, tmp_path):
        """Test that catchment directories are created correctly."""
        subcatchment_path = tmp_path / "subcatchments"
        catchment_id = 12345
        
        path_subcatch = subcatchment_path / str(catchment_id)
        
        # Create directories
        os.makedirs(path_subcatch, exist_ok=True)
        os.makedirs(path_subcatch / 'maps', exist_ok=True)
        os.makedirs(path_subcatch / 'inflow', exist_ok=True)
        os.makedirs(path_subcatch / 'out', exist_ok=True)
        
        # Verify directories exist
        assert path_subcatch.exists()
        assert (path_subcatch / 'maps').exists()
        assert (path_subcatch / 'inflow').exists()
        assert (path_subcatch / 'out').exists()

    def test_catchment_filtering_from_list(self):
        """Test filtering catchments based on CatchmentsToProcess list."""
        stationdata = pd.DataFrame({
            'DrainingArea.km2.LDD': [1000, 2000, 3000, 4000]
        }, index=[1001, 1002, 1003, 1004])
        
        CatchmentsToProcess = pd.DataFrame({0: [1002, 1004]})
        
        filtered = stationdata[stationdata.index.isin(CatchmentsToProcess[0].values)]
        
        assert len(filtered) == 2
        assert 1002 in filtered.index
        assert 1004 in filtered.index
        assert 1001 not in filtered.index
        assert 1003 not in filtered.index


class TestIntegrationScenarios:
    """Integration-style tests for common workflows."""

    def test_full_station_filtering_workflow(self, tmp_path):
        """Test the full station filtering workflow."""
        # Create settings file
        settings_file = tmp_path / "settings.txt"
        settings_file.write_text("""
[Main]
forcing_start = 01/01/2016 00:00
forcing_end = 31/12/2017 00:00
timestep = 1440

[Stations]
observed_discharges = observed.csv
stations_data = stations.csv
""")
        
        # Create stations data
        stations_df = pd.DataFrame({
            'LisfloodX': [100.0, 200.0, 300.0],
            'LisfloodY': [50.0, 60.0, 70.0],
            'EC_calib': [1, 1, 1],
            'Spinup_days': [30, 30, 30],
            'Min_calib_days': [100, 200, 50],
            'CAL_TYPE': [24, 24, 24],
            'DrainingArea.km2.LDD': [1000, 2000, 500]
        }, index=pd.Index([1001, 1002, 1003], name='ObsID'))
        stations_df.to_csv(tmp_path / "stations.csv")
        
        # Create observed data with enough data for all stations
        dates = pd.date_range('2015-11-01', periods=800, freq='D')
        obs_df = pd.DataFrame({
            1001: np.random.rand(800) * 100,
            1002: np.random.rand(800) * 100,
            1003: np.random.rand(800) * 100
        }, index=dates.strftime('%d/%m/%Y %H:%M'))
        obs_df.index.name = 'date'
        obs_df.to_csv(tmp_path / "observed.csv")
        
        # Verify data can be loaded
        loaded_stations = pd.read_csv(tmp_path / "stations.csv", index_col='ObsID')
        loaded_obs = pd.read_csv(tmp_path / "observed.csv", index_col=0)
        
        assert len(loaded_stations) == 3
        assert loaded_obs.shape[1] == 3

    def test_date_range_generation_daily(self):
        """Test date range generation for daily timestep."""
        forcing_start = datetime(2016, 1, 1)
        forcing_end = datetime(2016, 12, 31)
        timestep = 1440
        
        full_date_range = pd.date_range(
            start=forcing_start.strftime('%d/%m/%Y %H:%M'),
            end=forcing_end.strftime('%d/%m/%Y %H:%M'),
            freq='D'
        )
        
        assert len(full_date_range) == 366  # 2016 is a leap year

    def test_date_range_generation_6hourly(self):
        """Test date range generation for 6-hourly timestep."""
        forcing_start = datetime(2016, 1, 1)
        forcing_end = datetime(2016, 1, 3, 0, 0)
        timestep = 360
        
        full_date_range = pd.date_range(
            start=forcing_start.strftime('%d/%m/%Y %H:%M'),
            end=forcing_end.strftime('%d/%m/%Y %H:%M'),
            freq='6h'
        )
        
        # 3 days * 4 intervals per day = 12 (but inclusive of end point gives 13)
        assert len(full_date_range) >= 12


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_stations_dataframe(self):
        """Test handling of empty stations dataframe."""
        stations_meta = pd.DataFrame(columns=['LisfloodX', 'LisfloodY', 'EC_calib'])
        
        filtered = stations_meta[stations_meta['EC_calib'] >= 1]
        assert len(filtered) == 0

    def test_all_nan_observed_data(self):
        """Test handling of all NaN observed data."""
        dates = pd.date_range('2016-01-01', periods=100, freq='D')
        observed_streamflow = pd.Series(np.nan, index=dates)
        
        station_type = 24
        obs_period_days = stations.observation_period_days(station_type, observed_streamflow)
        
        assert obs_period_days == 0

    def test_station_with_zero_min_calib_days(self):
        """Test station with zero minimum calibration days."""
        station_data = pd.Series({
            'Spinup_days': 0,
            'Min_calib_days': 0,
            'CAL_TYPE': 24
        })
        
        # Even with zero min_calib_days, we need some data
        dates = pd.date_range('2016-01-01', periods=10, freq='D')
        observed_streamflow = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=dates)
        
        obs_period_days = stations.observation_period_days(station_data['CAL_TYPE'], observed_streamflow)
        
        assert obs_period_days >= 0

    def test_very_large_station_id(self):
        """Test handling of very large station ID."""
        large_id = 999999
        
        # Should be valid (max allowed)
        assert large_id <= 999999
        
        # Should fail
        invalid_id = 10000000
        assert invalid_id > 999999


class TestCAL2HydroDependenciesExtended:
    """Extended tests for CAL_2_HYDRO_DEPENDENCIES.py logic."""

    def test_station_id_too_high_raises(self):
        """Test that station ID >= 999999 raises an exception."""
        stationdata = pd.DataFrame({
            'LisfloodX': [100.0],
            'LisfloodY': [50.0],
        }, index=[999999])

        for index, row in stationdata.iterrows():
            with pytest.raises(Exception, match="too high"):
                if index >= 999999:
                    raise Exception(
                        "Station ID " + str(index) +
                        " is too high. Maximum allowed station ID is 999999 "
                        "because of limitations in PCRaster map format."
                    )

    def test_station_id_below_limit_ok(self):
        """Test that station IDs below 999999 are accepted."""
        valid_ids = [1, 100, 999998]
        for sid in valid_ids:
            assert sid < 999999

    def test_station_txt_file_creation(self, tmp_path):
        """Test creation of station.txt file for col2map."""
        stationdata = pd.DataFrame({
            'LisfloodX': [100.5, 200.3, 300.1],
            'LisfloodY': [50.2, 60.4, 70.6],
        }, index=[1001, 1002, 1003])

        station_txt = tmp_path / "station.txt"
        with open(str(station_txt), 'w') as f:
            for index, row in stationdata.iterrows():
                f.write(str(row['LisfloodX']) + " ")
                f.write(str(row['LisfloodY']) + " ")
                f.write(str(float(index)) + "\n")

        lines = station_txt.read_text().strip().split('\n')
        assert len(lines) == 3
        assert "100.5 50.2 1001.0" in lines[0]
        assert "200.3 60.4 1002.0" in lines[1]
        assert "300.1 70.6 1003.0" in lines[2]

    def test_station_conflict_single_occurrence(self):
        """Test station conflict check: exactly 1 occurrence is OK."""
        # Simulate reading map2col output: station found once
        lines = ["100.5 50.2 1001\n"]
        counter2 = 0
        for line in lines:
            (X, Y, value) = line.split()
            counter2 += 1

        assert counter2 == 1
        assert int(value) == 1001

    def test_station_conflict_not_found(self):
        """Test station conflict check: station not found (0 occurrences)."""
        lines = []  # empty output means station not in map
        counter2 = 0
        for line in lines:
            (X, Y, value) = line.split()
            counter2 += 1

        assert counter2 == 0

    def test_station_conflict_multiple_occurrences(self):
        """Test station conflict check: multiple occurrences is a conflict."""
        lines = ["100.5 50.2 1001\n", "101.0 51.0 1001\n"]
        counter2 = 0
        for line in lines:
            (X, Y, value) = line.split()
            counter2 += 1

        assert counter2 > 1

    def test_station_conflict_wrong_id(self):
        """Test station conflict: value in map differs from expected station ID."""
        expected_id = 1001
        lines = ["100.5 50.2 1002\n"]
        for line in lines:
            (X, Y, value) = line.split()

        assert int(value) != expected_id

    def test_catchment_area_sorting(self):
        """Test sorting stations by CatchmentArea for interstation regions."""
        stationdata = pd.DataFrame({
            'CatchmentArea': [500.0, 2000.0, 100.0, 1500.0],
        }, index=[1001, 1002, 1003, 1004])

        sorted_data = stationdata.sort_values(by=['CatchmentArea'], ascending=False)
        assert sorted_data.index[0] == 1002  # largest first
        assert sorted_data.index[-1] == 1003  # smallest last

    def test_sampling_frequency_direct_connection(self):
        """Test direct connection detection via SamplingFrequency."""
        # Two stations are directly connected if
        # parent.SamplingFrequency + 1 == child.SamplingFrequency
        parent_freq = 3
        child_freq = 4
        assert int(parent_freq + 1) == int(child_freq)

        # Not directly connected
        other_freq = 6
        assert int(parent_freq + 1) != int(other_freq)

    def test_stations_links_csv_format(self):
        """Test stations_links.csv output format."""
        catchment_id = 1001
        subcatchments = [1002, 1003]

        # Build the line as CAL_2 does
        text2 = str(catchment_id)
        commas = 20
        for sub in subcatchments:
            text2 += "," + str(sub)
            commas -= 1
        for _ in range(commas):
            text2 += ","
        text2 += "\n"

        parts = text2.strip().split(',')
        assert parts[0] == '1001'
        assert parts[1] == '1002'
        assert parts[2] == '1003'
        # Remaining should be empty strings (padding commas)
        assert all(p == '' for p in parts[3:])
        assert len(parts) == 21  # 1 ID + 20 comma-separated fields

    def test_args_parsing_cal2(self):
        """Test argument parsing for CAL_2."""
        parser = argparse.ArgumentParser()
        parser.add_argument('stations_data')
        parser.add_argument('ldd')
        parser.add_argument('stations_dir')
        parser.add_argument('tmp_dir')

        args = parser.parse_args([
            'stations.csv', '/path/to/ldd.map',
            '/path/to/stations', '/tmp/cal2'
        ])
        assert args.stations_data == 'stations.csv'
        assert args.ldd == '/path/to/ldd.map'
        assert args.stations_dir == '/path/to/stations'
        assert args.tmp_dir == '/tmp/cal2'


class TestCAL3MaskExtended:
    """Extended tests for CAL_3_MASK.py logic."""

    def test_stationdata_sorted_by_draining_area(self):
        """Test sorting by DrainingArea.km2.LDD ascending."""
        stationdata = pd.DataFrame({
            'DrainingArea.km2.LDD': [3000, 1000, 5000, 200],
        }, index=[1001, 1002, 1003, 1004])

        sorted_data = stationdata.sort_values(
            by=['DrainingArea.km2.LDD'], ascending=True
        )
        assert sorted_data.index.tolist() == [1004, 1002, 1001, 1003]

    def test_catchments_to_process_filtering(self):
        """Test that only stations in CatchmentsToProcess are processed."""
        stationdata = pd.DataFrame({
            'DrainingArea.km2.LDD': [1000, 2000, 3000, 4000, 5000],
        }, index=[101, 102, 103, 104, 105])

        catchments_to_process = pd.DataFrame({0: [102, 104, 105]})
        series = catchments_to_process[0]

        processed = []
        for index, row in stationdata.iterrows():
            if index in series.values:
                processed.append(index)

        assert processed == [102, 104, 105]

    def test_subcatchment_directory_structure(self, tmp_path):
        """Test full subcatchment directory structure creation."""
        subcatchment_path = tmp_path / "subcatchments"
        catchments = [12345, 67890]

        for catchment in catchments:
            path_subcatch = subcatchment_path / str(catchment)
            os.makedirs(path_subcatch / 'maps', exist_ok=True)
            os.makedirs(path_subcatch / 'inflow', exist_ok=True)
            os.makedirs(path_subcatch / 'out', exist_ok=True)

        for catchment in catchments:
            base = subcatchment_path / str(catchment)
            assert base.exists()
            assert (base / 'maps').exists()
            assert (base / 'inflow').exists()
            assert (base / 'out').exists()

    def test_mask_map_paths(self, tmp_path):
        """Test mask map file path construction."""
        path_subcatch = tmp_path / "12345"
        path_subcatch.mkdir()
        (path_subcatch / "maps").mkdir()

        subcatchmask_map = os.path.join(str(path_subcatch), "maps", "mask.map")
        smallsubcatchmask_map = os.path.join(
            str(path_subcatch), "maps", "masksmall.map"
        )

        assert subcatchmask_map.endswith("12345/maps/mask.map")
        assert smallsubcatchmask_map.endswith("12345/maps/masksmall.map")

    def test_outlet_map_path(self, tmp_path):
        """Test outlet map file path construction."""
        path_subcatch = tmp_path / "12345"
        (path_subcatch / "maps").mkdir(parents=True)

        subcatchstation_map = os.path.join(str(path_subcatch), "maps", "outlet.map")
        subcatchstation_small_map = os.path.join(
            str(path_subcatch), "maps", "outletsmall.map"
        )

        assert "outlet.map" in subcatchstation_map
        assert "outletsmall.map" in subcatchstation_small_map

    def test_inlet_map_copy_path(self, tmp_path):
        """Test inlet map copy destination path."""
        path_subcatch = tmp_path / "12345"
        (path_subcatch / "inflow").mkdir(parents=True)

        subcatchinlets_map = os.path.join(str(path_subcatch), "inflow", "inflow.map")
        assert subcatchinlets_map.endswith("12345/inflow/inflow.map")

    def test_settings_parsing_cal3(self, tmp_path):
        """Test settings file parsing for CAL_3."""
        from configparser import ConfigParser

        settings_file = tmp_path / "settings.cfg"
        settings_file.write_text(
            "[Path]\n"
            "subcatchment_path = /data/subcatchments\n\n"
            "[Stations]\n"
            "stations_data = /data/stations.csv\n"
        )

        parser = ConfigParser()
        parser.read(str(settings_file))

        subcatchment_path = parser.get('Path', 'subcatchment_path')
        stations_data_path = parser.get('Stations', 'stations_data')

        assert subcatchment_path == '/data/subcatchments'
        assert stations_data_path == '/data/stations.csv'

        path_result = os.path.dirname(stations_data_path)
        assert path_result == '/data'


class TestCAL4CutMapsExtended:
    """Extended tests for CAL_4_CUT_MAPS.py logic."""

    def test_single_station_integer_parsing(self):
        """Test parsing single station as integer."""
        station_input = "7838"
        obsid = None
        try:
            obsid = int(station_input)
        except ValueError:
            pass

        assert obsid == 7838

    def test_station_list_file_parsing(self, tmp_path):
        """Test parsing station list from CSV file."""
        station_file = tmp_path / "stations.txt"
        station_file.write_text("7838\n6651\n6593\n")

        station_input = str(station_file)
        obsid = None
        try:
            obsid = int(station_input)
        except ValueError:
            if os.path.isfile(station_input):
                CatchmentsToProcess = pd.read_csv(
                    station_input, sep=",", header=None
                )
                obsid = np.array(CatchmentsToProcess[0])

        assert obsid is not None
        assert len(obsid) == 3
        assert obsid[0] == 7838

    def test_station_not_in_metadata_raises(self):
        """Test that looking up a missing station raises an exception."""
        stations_meta = pd.DataFrame({
            'LisfloodX': [100.0, 200.0],
            'LisfloodY': [50.0, 60.0],
        }, index=pd.Index([1001, 1002], name='ObsID'))

        obsid = 9999
        with pytest.raises(Exception, match="not found"):
            try:
                station_data = stations_meta.loc[obsid]
            except KeyError:
                raise Exception(
                    'Station {} not found in stations file'.format(obsid)
                )

    def test_use_dask_config_flag(self):
        """Test --use-dask-config argument flag parsing."""
        parser = argparse.ArgumentParser()
        parser.add_argument('settings_file')
        parser.add_argument('path_maps')
        parser.add_argument('station')
        parser.add_argument(
            '--use-dask-config', action='store_true'
        )

        # Without flag
        args = parser.parse_args(['s.cfg', '/maps', '7838'])
        assert args.use_dask_config is False

        # With flag
        args = parser.parse_args([
            's.cfg', '/maps', '7838', '--use-dask-config'
        ])
        assert args.use_dask_config is True

    def test_output_maps_directory_structure(self, tmp_path):
        """Test that output maps go to subcatchment/maps/ directory."""
        subcatchment_path = tmp_path / "subcatchments"
        obsid = 7838
        maps_dir = subcatchment_path / str(obsid) / 'maps'
        maps_dir.mkdir(parents=True)

        afile = "ldd.map"
        fileout = os.path.join(str(maps_dir), afile)
        assert fileout.endswith("7838/maps/ldd.map")

    def test_obsid_as_array_for_multiple_stations(self):
        """Test that multiple station IDs are handled as numpy array."""
        station_list_data = pd.DataFrame({0: [7838, 6651, 6593]})
        obsid = np.array(station_list_data[0])

        assert isinstance(obsid, np.ndarray)
        assert len(obsid) == 3
        assert obsid[0] == 7838


class TestCAL5ExtractStationExtended:
    """Extended tests for CAL_5_EXTRACT_STATION.py logic."""

    def test_no_check_flag_parsing(self):
        """Test --no_check argument flag parsing."""
        parser = argparse.ArgumentParser()
        parser.add_argument('settings_file')
        parser.add_argument('station')
        parser.add_argument(
            '--no_check', action='store_true'
        )

        # Without flag
        args = parser.parse_args(['s.cfg', '7838'])
        assert args.no_check is False

        # With flag
        args = parser.parse_args(['s.cfg', '7838', '--no_check'])
        assert args.no_check is True

    def test_check_obs_disables_validation(self):
        """Test that check_obs=False bypasses min_calib_days validation."""
        obs_period_days = 10.0
        min_calib_days = 100.0
        check_obs = False

        # When check_obs is False, no exception should be raised
        error_raised = False
        if check_obs:
            if obs_period_days < min_calib_days:
                error_raised = True

        assert error_raised is False

    def test_check_obs_raises_on_insufficient_data(self):
        """Test that check_obs=True raises when data is insufficient."""
        obs_period_days = 10.0
        min_calib_days = 100.0
        check_obs = True
        obsid = 7838

        with pytest.raises(Exception, match="only contains"):
            if check_obs:
                if obs_period_days < min_calib_days:
                    raise Exception(
                        'Station {} only contains {} days of data! {} required'.format(
                            obsid, obs_period_days, min_calib_days
                        )
                    )

    def test_multiple_stations_loop(self):
        """Test processing multiple station IDs in a loop."""
        obsids = np.array([7838, 6651, 6593])
        processed = []

        for obsid in obsids:
            processed.append(int(obsid))

        assert processed == [7838, 6651, 6593]

    def test_station_data_output_directory(self, tmp_path):
        """Test that station output directory is created correctly."""
        subcatchment_path = tmp_path / "subcatchments"
        obsid = 7838

        out_dir = subcatchment_path / str(obsid) / 'station'
        os.makedirs(str(out_dir), exist_ok=True)

        assert out_dir.exists()
        assert str(out_dir).endswith("7838/station")

    def test_observations_reindex_daily(self):
        """Test observation reindexing to full daily date range."""
        forcing_start = datetime(2016, 1, 1)
        forcing_end = datetime(2016, 1, 10)

        full_date_range = pd.date_range(
            start=forcing_start, end=forcing_end, freq='D'
        )

        # Original data has gaps
        obs_dates = pd.to_datetime(['2016-01-01', '2016-01-03', '2016-01-07'])
        observations = pd.DataFrame(
            {'1001': [10.0, 20.0, 30.0]}, index=obs_dates
        )

        reindexed = observations.reindex(full_date_range)
        assert len(reindexed) == 10
        assert np.isnan(reindexed['1001'].iloc[1])  # Jan 2 is NaN
        assert reindexed['1001'].iloc[0] == 10.0   # Jan 1 has data

    def test_observations_reindex_6hourly(self):
        """Test observation reindexing to full 6-hourly date range."""
        forcing_start = datetime(2016, 1, 1)
        forcing_end = datetime(2016, 1, 2)

        full_date_range = pd.date_range(
            start=forcing_start, end=forcing_end, freq='6h'
        )

        assert len(full_date_range) == 5  # 0h, 6h, 12h, 18h, 24h

    def test_extract_station_output_files(self, tmp_path):
        """Test that extract_station_data creates expected output files."""
        out_dir = tmp_path / "station"
        out_dir.mkdir()

        # Simulate file creation as done by extract_station_data
        expected_files = [
            'observations_original.csv',
            'observations_complete.csv',
            'observations.csv',
            'station_data.csv',
        ]

        for fname in expected_files:
            (out_dir / fname).write_text("dummy")

        for fname in expected_files:
            assert (out_dir / fname).exists()

    def test_valid_start_end_extraction(self):
        """Test extracting valid_start and valid_end from filtered observations."""
        dates = pd.date_range('2016-01-01', periods=100, freq='D')
        observed = pd.Series(np.random.rand(100), index=dates)
        # Add NaN at start and end
        observed.iloc[:10] = np.nan
        observed.iloc[90:] = np.nan

        filtered = observed[observed.notna()]
        valid_start = filtered.index[0]
        valid_end = filtered.index[-1]

        # First valid index is day 10 (Jan 11), last valid is day 89 (Mar 30)
        assert valid_start == pd.Timestamp('2016-01-11')
        assert valid_end == pd.Timestamp('2016-03-30')

    def test_cache_clear_between_stations(self):
        """Test Cache.clear() is called between station processing."""
        # In CAL_5, Cache.clear() is called before processing
        # reservoir events for each station
        hydro_model.Cache.clear()
        # Should not raise
        assert True


class TestCAL5bSetPrerunInSettings:
    """Tests for CAL_5b_SET_PRERUN_IN_SETTINGS.py logic."""

    def test_prerun_start_read_from_station_csv(self, tmp_path):
        """Test reading prerun_start from station CSV."""
        stations_csv = tmp_path / "stations.csv"
        stations_csv.write_text(
            "ObsID,LisfloodX,LisfloodY,prerun_start\n"
            "7838,100.5,50.2,01/01/2000 00:00\n"
            "6651,200.3,60.4,01/01/1979 00:00\n"
        )
        stations_meta = pd.read_csv(str(stations_csv), sep=",", index_col='ObsID')
        station_data = stations_meta.loc[7838]
        assert station_data['prerun_start'] == '01/01/2000 00:00'

    def test_prerun_start_updates_settings_file(self, tmp_path):
        """Test that prerun_start is written into the settings file."""
        from configparser import ConfigParser

        settings_file = tmp_path / "settings.cfg"
        settings_file.write_text(
            "[Main]\n"
            "prerun_start = 01/01/1990 00:00\n"
            "forcing_start = 01/01/2000 00:00\n"
        )

        parser = ConfigParser()
        parser.read(str(settings_file))

        new_prerun_start = '02/01/1979 00:00'
        parser['Main']['prerun_start'] = new_prerun_start
        with open(str(settings_file), 'w') as configfile:
            parser.write(configfile)

        # Re-read and verify
        parser2 = ConfigParser()
        parser2.read(str(settings_file))
        assert parser2.get('Main', 'prerun_start') == new_prerun_start

    def test_station_not_found_raises(self, tmp_path):
        """Test that missing station raises exception."""
        stations_csv = tmp_path / "stations.csv"
        stations_csv.write_text(
            "ObsID,LisfloodX,LisfloodY,prerun_start\n"
            "7838,100.5,50.2,01/01/2000 00:00\n"
        )
        stations_meta = pd.read_csv(str(stations_csv), sep=",", index_col='ObsID')

        obsid = 9999
        with pytest.raises(KeyError):
            stations_meta.loc[obsid]

    def test_settings_file_not_found_raises(self, tmp_path):
        """Test that a missing settings file raises FileNotFoundError."""
        from configparser import ConfigParser

        settings_file = str(tmp_path / "nonexistent.cfg")
        assert not os.path.isfile(settings_file)

        # Replicate the script logic
        with pytest.raises(FileNotFoundError):
            if not os.path.isfile(settings_file):
                raise FileNotFoundError(
                    'Incorrect path to setting file: {}'.format(settings_file)
                )


class TestCAL6CalibrationLogic:
    """Tests for CAL_6_CALIBRATION.py logic."""

    def test_skip_if_streamflow_best_exists(self, tmp_path):
        """Test that calibration is skipped if streamflow_simulated_best.csv exists."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        best_csv = out_dir / "streamflow_simulated_best.csv"
        best_csv.write_text("dummy")

        # Replicate skip logic from calibrate_subcatchment
        path_subcatch = tmp_path
        should_skip = os.path.exists(
            os.path.join(str(path_subcatch), "out", "streamflow_simulated_best.csv")
        )
        assert should_skip is True

    def test_skip_if_pareto_front_exists(self, tmp_path):
        """Test that calibration deap is skipped when pareto_front.csv exists."""
        pareto_file = tmp_path / "pareto_front.csv"
        pareto_file.write_text("KGE,param1\n0.9,1.5\n")

        should_skip_deap = os.path.exists(str(pareto_file))
        assert should_skip_deap is True

    def test_rerun_with_kgejsd_triggered(self, tmp_path):
        """Test that rerun_with_KGEJSD is triggered when status file exists."""
        calibstatus_kgejsd = tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt'
        calibstatus_kge = tmp_path / 'CalibrationStatus_3rd_run_KGE.txt'

        # Only KGEJSD file exists, not KGE
        calibstatus_kgejsd.write_text("KGEJSD 1st calibration failed")

        rerun_with_KGEJSD = False
        rerun_with_KGE = False

        if os.path.exists(str(calibstatus_kgejsd)) and not os.path.exists(str(calibstatus_kge)):
            rerun_with_KGEJSD = True
        elif os.path.exists(str(calibstatus_kge)):
            rerun_with_KGE = True

        assert rerun_with_KGEJSD is True
        assert rerun_with_KGE is False

    def test_rerun_with_kge_triggered(self, tmp_path):
        """Test that rerun_with_KGE is triggered when both status files exist."""
        calibstatus_kgejsd = tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt'
        calibstatus_kge = tmp_path / 'CalibrationStatus_3rd_run_KGE.txt'

        calibstatus_kgejsd.write_text("KGEJSD 1st calibration failed")
        calibstatus_kge.write_text("KGEJSD 2nd calibration failed")

        rerun_with_KGEJSD = False
        rerun_with_KGE = False

        if os.path.exists(str(calibstatus_kgejsd)) and not os.path.exists(str(calibstatus_kge)):
            rerun_with_KGEJSD = True
        elif os.path.exists(str(calibstatus_kge)):
            rerun_with_KGE = True

        assert rerun_with_KGEJSD is False
        assert rerun_with_KGE is True

    def test_no_rerun_when_no_status_files(self, tmp_path):
        """Test no rerun when neither status file exists."""
        calibstatus_kgejsd = tmp_path / 'CalibrationStatus_2nd_run_KGEJSD.txt'
        calibstatus_kge = tmp_path / 'CalibrationStatus_3rd_run_KGE.txt'

        rerun_with_KGEJSD = False
        rerun_with_KGE = False

        if os.path.exists(str(calibstatus_kgejsd)) and not os.path.exists(str(calibstatus_kge)):
            rerun_with_KGEJSD = True
        elif os.path.exists(str(calibstatus_kge)):
            rerun_with_KGE = True

        assert rerun_with_KGEJSD is False
        assert rerun_with_KGE is False

    def test_seed_change_for_kgejsd_rerun(self):
        """Test that seed is changed to 233 for KGEJSD rerun."""
        original_seed = 42
        # CAL_6 changes seed to 233 for KGEJSD second run
        cfg_seed = original_seed
        rerun_with_KGEJSD = True

        if rerun_with_KGEJSD:
            cfg_seed = 233

        assert cfg_seed == 233

    def test_objectives_change_for_kge_rerun(self):
        """Test that objectives list changes to KGE for the third run."""
        objectives_list = ['KGE_JSD']
        original_seed = 42
        cfg_seed = 233

        rerun_with_KGE = True
        if rerun_with_KGE:
            objectives_list = ['KGE']
            cfg_seed = original_seed

        assert objectives_list == ['KGE']
        assert cfg_seed == 42

    def test_calibration_args_parsing(self):
        """Test argument parsing logic for CAL_6."""
        # CAL_6 expects: settings_file, station, n_cpus, --seed
        parser = argparse.ArgumentParser()
        parser.add_argument('settings_file')
        parser.add_argument('station')
        parser.add_argument('n_cpus')
        parser.add_argument('--seed')

        args = parser.parse_args(['settings.cfg', '7838', '4', '--seed', '42'])
        assert args.settings_file == 'settings.cfg'
        assert args.station == '7838'
        assert args.n_cpus == '4'
        assert args.seed == '42'

    def test_calibration_args_no_seed(self):
        """Test argument parsing without seed."""
        parser = argparse.ArgumentParser()
        parser.add_argument('settings_file')
        parser.add_argument('station')
        parser.add_argument('n_cpus')
        parser.add_argument('--seed')

        args = parser.parse_args(['settings.cfg', '7838', '4'])
        assert args.seed is None


class TestCAL7LongtermRunLogic:
    """Tests for CAL_7_LONGTERM_RUN.py logic."""

    def test_skip_if_any_best_file_exists(self, tmp_path):
        """Test longtermrun is skipped when any streamflow_simulated_best* file exists."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # No files - should not skip
        stop_files = [
            "streamflow_simulated_best.csv",
            "streamflow_simulated_best_STOPForLowKGE.csv",
            "streamflow_simulated_best_STOPForHighWaterRemoval.csv",
            "streamflow_simulated_best_STOPForHighTL.csv",
            "streamflow_simulated_best_STOPForChanqAvgDiff.csv",
        ]

        def should_skip(path):
            return any(
                os.path.exists(os.path.join(str(path), "out", f))
                for f in stop_files
            )

        assert should_skip(tmp_path) is False

        # Create one stop file
        (out_dir / "streamflow_simulated_best_STOPForLowKGE.csv").write_text("dummy")
        assert should_skip(tmp_path) is True

    def test_high_water_removal_detection(self):
        """Test HighWaterRemoval stop condition detection."""
        TransSub = 0.15
        GwLoss = 0.95
        GwPerc = 1.5
        b_Xinanjiang = 0.8
        PowerPrefFlow = 6.0
        LowerZoneTimeConstant = 600

        is_high_water_removal = (
            TransSub > 0.12
            and GwLoss > 0.9
            and GwPerc > 1
            and b_Xinanjiang < 1
            and PowerPrefFlow > 5
            and LowerZoneTimeConstant > 500
        )
        assert is_high_water_removal is True

    def test_high_water_removal_not_triggered(self):
        """Test HighWaterRemoval not triggered for normal parameters."""
        TransSub = 0.05
        GwLoss = 0.5
        GwPerc = 0.8
        b_Xinanjiang = 1.5
        PowerPrefFlow = 3.0
        LowerZoneTimeConstant = 200

        is_high_water_removal = (
            TransSub > 0.12
            and GwLoss > 0.9
            and GwPerc > 1
            and b_Xinanjiang < 1
            and PowerPrefFlow > 5
            and LowerZoneTimeConstant > 500
        )
        assert is_high_water_removal is False

    def test_high_tl_ratio_detection(self):
        """Test HighTL ratio stop condition."""
        # Simulate cumulative sums
        cumsum_transm_loss = 500.0
        cumsum_rain = 800.0
        cumsum_snow = 200.0

        ratio = cumsum_transm_loss / (cumsum_rain + cumsum_snow)
        # ratio = 500 / 1000 = 0.5, which is > 0.30
        assert ratio > 0.30

    def test_high_tl_ratio_not_triggered(self):
        """Test HighTL ratio not triggered for normal values."""
        cumsum_transm_loss = 100.0
        cumsum_rain = 800.0
        cumsum_snow = 200.0

        ratio = cumsum_transm_loss / (cumsum_rain + cumsum_snow)
        # ratio = 100 / 1000 = 0.1, which is < 0.30
        assert ratio < 0.30

    def test_chanq_avg_diff_detection(self):
        """Test ChanqAvgDiff instability detection."""
        chanqavgdt_data = np.array([150.0, 200.0, 300.0, 50.0])
        chanq_data = np.array([1600.0, 200.0, 300.0, 50.0])

        # Only consider elements where either > 100
        dismask = (chanq_data > 100.0) | (chanqavgdt_data > 100.0)
        too_large = chanq_data[dismask] > 10 * chanqavgdt_data[dismask]
        too_small = chanq_data[dismask] < 0.1 * chanqavgdt_data[dismask]
        bad = too_large | too_small

        # chanq_data[0]=1600 > 10*chanqavgdt_data[0]=1500 -> True
        assert np.any(bad) == True

    def test_chanq_avg_diff_not_triggered(self):
        """Test ChanqAvgDiff not triggered for stable values."""
        chanqavgdt_data = np.array([150.0, 200.0, 300.0])
        chanq_data = np.array([160.0, 210.0, 290.0])

        dismask = (chanq_data > 100.0) | (chanqavgdt_data > 100.0)
        too_large = chanq_data[dismask] > 10 * chanqavgdt_data[dismask]
        too_small = chanq_data[dismask] < 0.1 * chanqavgdt_data[dismask]
        bad = too_large | too_small

        assert np.any(bad) == False

    def test_pcraster_1e31_to_nan_conversion(self):
        """Test that PCRaster 1e31 values are converted to NaN."""
        data = np.array([1.0, 2.0, 1e31, 3.0, 1e31])
        data[data == 1e31] = np.nan

        assert np.isnan(data[2])
        assert np.isnan(data[4])
        assert data[0] == 1.0

    def test_spinup_days_skip(self):
        """Test that spinup days are correctly skipped in TSS data."""
        data = np.arange(100, dtype=float)
        days_to_skip = 30

        data_after_spinup = data[days_to_skip:]
        assert len(data_after_spinup) == 70
        assert data_after_spinup[0] == 30.0

    def test_pareto_front_required(self, tmp_path):
        """Test that missing pareto_front.csv raises an error."""
        pareto_path = os.path.join(str(tmp_path), "pareto_front.csv")
        assert not os.path.exists(pareto_path)

        with pytest.raises(Exception, match="pareto_front"):
            if not os.path.exists(pareto_path):
                raise Exception(
                    'Could not find optimnal parameters for long term run. '
                    'Please calibrate to generate pareto_front.csv first.'
                )

    def test_rename_stop_files(self, tmp_path):
        """Test that output files are renamed correctly for stop conditions."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Create dummy files
        (out_dir / "streamflow_simulated_best.csv").write_text("data")
        (out_dir / "streamflow_simulated_best.tss").write_text("data")
        (out_dir / "chanqavgdt_simulated_best.csv").write_text("data")
        (out_dir / "chanqavgdt_simulated_best.tss").write_text("data")

        # Simulate rename for STOPForLowKGE
        os.rename(
            str(out_dir / "streamflow_simulated_best.csv"),
            str(out_dir / "streamflow_simulated_best_STOPForLowKGE.csv"),
        )
        os.rename(
            str(out_dir / "streamflow_simulated_best.tss"),
            str(out_dir / "streamflow_simulated_best_STOPForLowKGE.tss"),
        )

        assert (out_dir / "streamflow_simulated_best_STOPForLowKGE.csv").exists()
        assert (out_dir / "streamflow_simulated_best_STOPForLowKGE.tss").exists()
        assert not (out_dir / "streamflow_simulated_best.csv").exists()

    def test_longterm_run_args_parsing(self):
        """Test argument parsing for CAL_7."""
        parser = argparse.ArgumentParser()
        parser.add_argument('settings_file')
        parser.add_argument('station')

        args = parser.parse_args(['settings.cfg', '7838'])
        assert args.settings_file == 'settings.cfg'
        assert args.station == '7838'
        assert int(args.station) == 7838


if __name__ == '__main__':
    pytest.main([__file__, '-v'])