"""
Integration test for CAL_3_MASK.

Runs the full script on synthetic data (reusing CAL_2 reference maps)
and verifies that the correct subcatchment directory structure and
mask maps are produced.

Test data lives in tests/data/CAL_3/ and is self-contained:
- input/settings.txt               Settings template
- input/CatchmentsToProcess.txt    Station IDs to process (100, 200, 300)
- input/stations/                   CAL_2 output maps + stations_data.csv
- reference/catchments/             Expected output structure
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
CAL3_DATA = os.path.join(TEST_DIR, 'data', 'CAL_3')
INPUT_DIR = os.path.join(CAL3_DATA, 'input')
REF_DIR = os.path.join(CAL3_DATA, 'reference', 'catchments')


def _create_settings(tmp_path):
    """Create resolved settings pointing subcatchment_path to tmp output."""
    template = open(os.path.join(INPUT_DIR, 'settings.txt')).read()
    subcatchment_path = str(tmp_path / 'catchments')
    content = template.replace('{SUBCATCHMENT_PATH}', subcatchment_path)
    content = content.replace('{INPUT_DIR}', INPUT_DIR)
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)
    return settings_file, subcatchment_path


class TestCAL3Mask:
    """Integration tests for CAL_3_MASK main()."""

    @pytest.fixture
    def run_cal3(self, tmp_path):
        """Run CAL_3 main() and return the subcatchment output path."""
        from CAL_3_MASK import main

        settings_file, subcatchment_path = _create_settings(tmp_path)
        catchments_file = os.path.join(INPUT_DIR, 'CatchmentsToProcess.txt')

        main(settings_file, catchments_file)
        return subcatchment_path

    def test_directories_created_for_all_stations(self, run_cal3):
        """A subcatchment directory with maps/inflow/out is created for each station."""
        for station_id in [100, 200, 300]:
            base = os.path.join(run_cal3, str(station_id))
            assert os.path.isdir(base), f"Missing dir for station {station_id}"
            assert os.path.isdir(os.path.join(base, 'maps'))
            assert os.path.isdir(os.path.join(base, 'inflow'))
            assert os.path.isdir(os.path.join(base, 'out'))

    def test_mask_maps_created(self, run_cal3):
        """mask.map and masksmall.map are produced for each station."""
        for station_id in [100, 200, 300]:
            maps_dir = os.path.join(run_cal3, str(station_id), 'maps')
            assert os.path.isfile(os.path.join(maps_dir, 'mask.map'))
            assert os.path.isfile(os.path.join(maps_dir, 'masksmall.map'))

    def test_outlet_maps_created(self, run_cal3):
        """outlet.map and outletsmall.map are produced for each station."""
        for station_id in [100, 200, 300]:
            maps_dir = os.path.join(run_cal3, str(station_id), 'maps')
            assert os.path.isfile(os.path.join(maps_dir, 'outlet.map'))
            assert os.path.isfile(os.path.join(maps_dir, 'outletsmall.map'))

    def test_inflow_map_created(self, run_cal3):
        """inflow/inflow.map is produced for each station."""
        for station_id in [100, 200, 300]:
            inflow_dir = os.path.join(run_cal3, str(station_id), 'inflow')
            assert os.path.isfile(os.path.join(inflow_dir, 'inflow.map'))

    def test_mask_maps_match_reference(self, run_cal3):
        """mask.map binary content matches the reference for each station."""
        for station_id in [100, 200, 300]:
            result = open(os.path.join(run_cal3, str(station_id), 'maps', 'mask.map'), 'rb').read()
            ref = open(os.path.join(REF_DIR, str(station_id), 'maps', 'mask.map'), 'rb').read()
            assert result == ref, f"mask.map mismatch for station {station_id}"

    def test_masksmall_maps_match_reference(self, run_cal3):
        """masksmall.map binary content matches the reference for each station."""
        for station_id in [100, 200, 300]:
            result = open(os.path.join(run_cal3, str(station_id), 'maps', 'masksmall.map'), 'rb').read()
            ref = open(os.path.join(REF_DIR, str(station_id), 'maps', 'masksmall.map'), 'rb').read()
            assert result == ref, f"masksmall.map mismatch for station {station_id}"

    def test_subset_of_catchments_processed(self, tmp_path):
        """Only stations listed in CatchmentsToProcess are processed."""
        from CAL_3_MASK import main

        settings_file, subcatchment_path = _create_settings(tmp_path)

        # Create a CatchmentsToProcess with only station 200
        partial_file = str(tmp_path / 'partial.txt')
        with open(partial_file, 'w') as f:
            f.write("200\n")

        main(settings_file, partial_file)

        # Only station 200 should be created
        assert os.path.isdir(os.path.join(subcatchment_path, '200'))
        assert not os.path.isdir(os.path.join(subcatchment_path, '100'))
        assert not os.path.isdir(os.path.join(subcatchment_path, '300'))
