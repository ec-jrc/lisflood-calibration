"""
Integration test for CAL_1_FILTER_STATIONS.

Runs the full script on a subset of real Australia GloFASv5 data (5 stations,
3 valid + 2 invalid) and compares the output against reference files.

The test data lives in tests/data/CAL_1/ and is self-contained:
- input/settings.txt               Settings template (placeholders for paths)
- input/stations_data_subset.csv   5-station metadata
- input/observations_subset.csv    Observations (2010–2023, ~200KB)
- reference/stations_data.csv      Expected valid stations output
- reference/stations_data_invalid.csv  Expected invalid stations output
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
CAL1_DATA = os.path.join(TEST_DIR, 'data', 'CAL_1')
INPUT_DIR = os.path.join(CAL1_DATA, 'input')
REF_DIR = os.path.join(CAL1_DATA, 'reference')


def _create_settings(tmp_path):
    """Create a resolved settings file pointing to the test input and tmp output."""
    template = open(os.path.join(INPUT_DIR, 'settings.txt')).read()
    content = template.replace('{INPUT_DIR}', INPUT_DIR)
    content = content.replace('{OUTPUT_DIR}', str(tmp_path))
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)
    return settings_file


class TestCAL1FilterStations:
    """Integration tests for CAL_1_FILTER_STATIONS main()."""

    def test_valid_stations_match_reference(self, tmp_path):
        """The set of valid station IDs matches the reference output."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        valid_df, invalid_df = main(settings_file, stations_csv, '1')

        ref_valid = pd.read_csv(os.path.join(REF_DIR, 'stations_data.csv'), index_col='ObsID')
        assert set(valid_df.index) == set(ref_valid.index)

    def test_invalid_stations_match_reference(self, tmp_path):
        """The set of invalid station IDs matches the reference output."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        valid_df, invalid_df = main(settings_file, stations_csv, '1')

        ref_invalid = pd.read_csv(os.path.join(REF_DIR, 'stations_data_invalid.csv'), index_col='ObsID')
        assert set(invalid_df.index) == set(ref_invalid.index)

    def test_three_valid_two_invalid(self, tmp_path):
        """Exactly 3 stations are valid and 2 are invalid."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        valid_df, invalid_df = main(settings_file, stations_csv, '1')

        assert len(valid_df) == 3
        assert len(invalid_df) == 2

    def test_output_csv_files_written(self, tmp_path):
        """main() writes stations_data.csv and stations_data_invalid.csv to disk."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        main(settings_file, stations_csv, '1')

        assert os.path.isfile(str(tmp_path / 'stations_data.csv'))
        assert os.path.isfile(str(tmp_path / 'stations_data_invalid.csv'))

    def test_written_valid_csv_matches_returned_dataframe(self, tmp_path):
        """The valid CSV written to disk matches the returned DataFrame."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        valid_df, _ = main(settings_file, stations_csv, '1')

        written_df = pd.read_csv(str(tmp_path / 'stations_data.csv'), index_col='ObsID')
        pd.testing.assert_frame_equal(valid_df, written_df)

    def test_written_invalid_csv_matches_reference(self, tmp_path):
        """The written stations_data_invalid.csv content matches the reference."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        _, invalid_df = main(settings_file, stations_csv, '1')

        written_invalid = pd.read_csv(str(tmp_path / 'stations_data_invalid.csv'), index_col='ObsID')
        ref_invalid = pd.read_csv(os.path.join(REF_DIR, 'stations_data_invalid.csv'), index_col='ObsID')

        pd.testing.assert_frame_equal(written_invalid, ref_invalid)
        pd.testing.assert_frame_equal(invalid_df, ref_invalid)

    def test_valid_station_columns_preserved(self, tmp_path):
        """All original metadata columns are preserved in the valid output."""
        from CAL_1_FILTER_STATIONS import main

        settings_file = _create_settings(tmp_path)
        stations_csv = os.path.join(INPUT_DIR, 'stations_data_subset.csv')

        valid_df, _ = main(settings_file, stations_csv, '1')

        ref_valid = pd.read_csv(os.path.join(REF_DIR, 'stations_data.csv'), index_col='ObsID')
        assert list(valid_df.columns) == list(ref_valid.columns)
