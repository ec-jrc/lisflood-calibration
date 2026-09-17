"""
Integration test for CAL_5_EXTRACT_STATION.

Runs the full script on synthetic observation data (2 stations, 2555 days)
and verifies that station extraction produces the expected output files
with correct content (observation periods, split dates, etc.).

Test data lives in tests/data/CAL_5/:
- input/settings.txt           ConfigCalibration-compatible settings template
- input/observations.csv       Synthetic daily observations for 2 stations
- input/stations_data.csv      Station metadata
- input/param_ranges.csv       Minimal parameter ranges (stub)
- input/lisflood_template.xml  Stub template
- input/stations_links.csv     Stub links
- reference/catchments/100/station/  Expected output for station 100
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
CAL5_DATA = os.path.join(TEST_DIR, 'data', 'CAL_5')
INPUT_DIR = os.path.join(CAL5_DATA, 'input')
REF_DIR = os.path.join(CAL5_DATA, 'reference', 'catchments')


def _create_settings(tmp_path):
    """Create resolved settings pointing subcatchment_path to tmp."""
    template = open(os.path.join(INPUT_DIR, 'settings.txt')).read()
    subcatch_path = str(tmp_path / 'catchments')
    content = template.replace('{SUBCATCHMENT_PATH}', subcatch_path)
    content = content.replace('{INPUT_DIR}', INPUT_DIR)
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)
    return settings_file, subcatch_path


class TestCAL5ExtractStation:
    """Integration tests for CAL_5_EXTRACT_STATION main()."""

    @pytest.fixture
    def run_cal5(self, tmp_path):
        """Run CAL_5 for station 100 and return the output path."""
        from CAL_5_EXTRACT_STATION import main

        settings_file, subcatch_path = _create_settings(tmp_path)
        main(settings_file, '100')
        return os.path.join(subcatch_path, '100', 'station')

    def test_output_files_created(self, run_cal5):
        """All expected output CSV files are created."""
        expected = [
            'observations_original.csv',
            'observations_complete.csv',
            'observations.csv',
            'station_data.csv',
        ]
        for fname in expected:
            assert os.path.isfile(os.path.join(run_cal5, fname)), f"Missing: {fname}"

    def test_station_data_contains_expected_fields(self, run_cal5):
        """station_data.csv includes Obs_start, Obs_end, Split_date, N_data."""
        df = pd.read_csv(os.path.join(run_cal5, 'station_data.csv'), index_col=0)
        for field in ['Obs_start', 'Obs_end', 'Split_date', 'N_data']:
            assert field in df.index, f"Missing field: {field}"

    def test_station_data_matches_reference(self, run_cal5):
        """station_data.csv content matches reference."""
        result = pd.read_csv(os.path.join(run_cal5, 'station_data.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(REF_DIR, '100', 'station', 'station_data.csv'), index_col=0)
        pd.testing.assert_frame_equal(result, ref)

    def test_observations_csv_matches_reference(self, run_cal5):
        """observations.csv (filtered valid period) matches reference."""
        result = pd.read_csv(os.path.join(run_cal5, 'observations.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(REF_DIR, '100', 'station', 'observations.csv'), index_col=0)
        pd.testing.assert_frame_equal(result, ref)

    def test_n_data_equals_observation_rows(self, run_cal5):
        """N_data in station_data.csv matches the number of rows in observations.csv."""
        station_data = pd.read_csv(os.path.join(run_cal5, 'station_data.csv'), index_col=0)
        obs = pd.read_csv(os.path.join(run_cal5, 'observations.csv'), index_col=0)
        n_data = int(float(station_data.loc['N_data'].values[0]))
        # N_data counts non-NaN entries in the valid period
        assert n_data == obs.iloc[:, 0].notna().sum()

    def test_no_check_allows_insufficient_data(self, tmp_path):
        """With no_check=True, stations with insufficient data are still processed."""
        from CAL_5_EXTRACT_STATION import main

        settings_file, subcatch_path = _create_settings(tmp_path)
        # Station 200 has enough data (same as 100), so this should work regardless.
        # But we verify no_check=True doesn't error.
        main(settings_file, '200', no_check=True)

        station_dir = os.path.join(subcatch_path, '200', 'station')
        assert os.path.isfile(os.path.join(station_dir, 'observations.csv'))

    def test_multiple_stations_via_list_file(self, tmp_path):
        """Processing a station list file extracts data for all listed stations."""
        from CAL_5_EXTRACT_STATION import main

        settings_file, subcatch_path = _create_settings(tmp_path)

        station_list = str(tmp_path / 'stations.txt')
        with open(station_list, 'w') as f:
            f.write("100\n200\n")

        main(settings_file, station_list)

        assert os.path.isfile(os.path.join(subcatch_path, '100', 'station', 'observations.csv'))
        assert os.path.isfile(os.path.join(subcatch_path, '200', 'station', 'observations.csv'))


# --- CAL_5 with reservoir_events (uses real 7838 data from CAL_5c_6_7) ---

import shutil

CAL5c67_DATA = os.path.join(TEST_DIR, 'data', 'CAL_5c_6_7')
CAL5c67_INPUT = os.path.join(CAL5c67_DATA, 'input')
CAL5c67_REF = os.path.join(CAL5c67_DATA, 'reference')
CAL5c67_TABLES = os.path.join(CAL5c67_INPUT, 'catchments', '7838', 'tables')


def _create_settings_with_reservoirs(tmp_path):
    """Create resolved settings with reservoir_events for station 7838."""
    tmp_catchments = str(tmp_path / 'catchments')
    shutil.copytree(os.path.join(CAL5c67_INPUT, 'catchments'), tmp_catchments)

    # Resolve XML template
    xml_src = os.path.join(CAL5c67_INPUT, 'templates', 'OSLisfloodGloFASv5calibration_v1.xml')
    xml_dst = str(tmp_path / 'template.xml')
    xml_content = open(xml_src).read()
    xml_content = xml_content.replace('{TABLES_DIR}/', CAL5c67_TABLES + '/')
    with open(xml_dst, 'w') as f:
        f.write(xml_content)

    # Resolve settings WITH reservoir_events
    template = open(os.path.join(CAL5c67_INPUT, 'settings_with_reservoirs.txt')).read()
    content = template.replace('{INPUT_DIR}', CAL5c67_INPUT)
    content = content.replace(
        f'subcatchment_path = {CAL5c67_INPUT}/catchments',
        f'subcatchment_path = {tmp_catchments}',
    )
    content = content.replace(
        f'LISFLOODSettings = {CAL5c67_INPUT}/templates/OSLisfloodGloFASv5calibration_v1.xml',
        f'LISFLOODSettings = {xml_dst}',
    )
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)

    return settings_file, tmp_catchments


class TestCAL5WithReservoirEvents:
    """Integration test for CAL_5 with reservoir_events creating filtered observations."""

    def test_extract_station_with_reservoir_filtering(self, tmp_path):
        """Run CAL_5 with reservoir_events and verify observations are filtered."""
        from CAL_5_EXTRACT_STATION import main

        settings_file, tmp_catchments = _create_settings_with_reservoirs(tmp_path)
        main(settings_file, '7838')

        station_dir = os.path.join(tmp_catchments, '7838', 'station')

        # Check output files
        assert os.path.isfile(os.path.join(station_dir, 'observations.csv'))
        assert os.path.isfile(os.path.join(station_dir, 'station_data.csv'))

        # FilteredReservoirMap.nc should be created (reservoir is active)
        assert os.path.isfile(os.path.join(station_dir, 'FilteredReservoirMap.nc')), \
            "Missing FilteredReservoirMap.nc — reservoir filtering did not run"

        # Obs_start should be adjusted due to reservoir event (2018 construction)
        result = pd.read_csv(os.path.join(station_dir, 'station_data.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(CAL5c67_REF, 'station_data_after_cal5_reservoir.csv'), index_col=0)

        # Compare Obs_start/Obs_end/Split_date
        assert result.loc['Obs_start'].values[0] == ref.loc['Obs_start'].values[0]
        assert result.loc['Obs_end'].values[0] == ref.loc['Obs_end'].values[0]
        assert int(float(result.loc['N_data'].values[0])) == int(float(ref.loc['N_data'].values[0]))
