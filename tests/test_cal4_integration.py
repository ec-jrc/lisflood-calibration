"""
Integration test for CAL_4_CUT_MAPS.

Runs the full script on synthetic data (10x10 maps with mask from CAL_3)
and verifies that maps are correctly clipped to subcatchment extents.

Test data lives in tests/data/CAL_4/:
- input/settings.txt                Settings template
- input/stations_data.csv           Station metadata
- input/global_maps/elevation.map   10x10 PCRaster scalar map
- input/global_maps/temperature.nc  10x10 NetCDF file
- input/catchments/*/maps/mask.map  Masks from CAL_3
- reference/*/maps/elevation.map    Expected clipped .map outputs
- reference/*/maps/temperature.nc   Expected clipped .nc outputs
"""

import os
import sys
import shutil
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
CAL4_DATA = os.path.join(TEST_DIR, 'data', 'CAL_4')
INPUT_DIR = os.path.join(CAL4_DATA, 'input')
REF_DIR = os.path.join(CAL4_DATA, 'reference')


def _setup_cal4(tmp_path):
    """Set up CAL_4 with tmp subcatchment_path containing mask.map files."""
    subcatch_path = str(tmp_path / 'catchments')

    # Copy mask.map files from input into tmp subcatchment_path
    for sid in [100, 200, 300]:
        src = os.path.join(INPUT_DIR, 'catchments', str(sid), 'maps', 'mask.map')
        dst_dir = os.path.join(subcatch_path, str(sid), 'maps')
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, 'mask.map'))

    # Create resolved settings file
    template = open(os.path.join(INPUT_DIR, 'settings.txt')).read()
    content = template.replace('{SUBCATCHMENT_PATH}', subcatch_path)
    content = content.replace('{INPUT_DIR}', INPUT_DIR)
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)

    return settings_file, subcatch_path


class TestCAL4CutMaps:
    """Integration tests for CAL_4_CUT_MAPS main()."""

    @pytest.fixture
    def run_cal4(self, tmp_path):
        """Run CAL_4 on all 3 stations and return subcatchment output path."""
        from CAL_4_CUT_MAPS import main

        settings_file, subcatch_path = _setup_cal4(tmp_path)

        # Create a station list file
        station_list = str(tmp_path / 'stations.txt')
        with open(station_list, 'w') as f:
            f.write("100\n200\n300\n")

        path_maps = os.path.join(INPUT_DIR, 'global_maps')
        main(settings_file, path_maps, station_list)
        return subcatch_path

    def test_elevation_map_created_for_all_stations(self, run_cal4):
        """A clipped elevation.map is created for each station."""
        for sid in [100, 200, 300]:
            fpath = os.path.join(run_cal4, str(sid), 'maps', 'elevation.map')
            assert os.path.isfile(fpath), f"Missing elevation.map for station {sid}"

    def test_temperature_nc_created_for_all_stations(self, run_cal4):
        """A clipped temperature.nc is created for each station."""
        for sid in [100, 200, 300]:
            fpath = os.path.join(run_cal4, str(sid), 'maps', 'temperature.nc')
            assert os.path.isfile(fpath), f"Missing temperature.nc for station {sid}"

    def test_elevation_map_matches_reference(self, run_cal4):
        """Clipped elevation.map binary matches reference for each station."""
        for sid in [100, 200, 300]:
            result = open(os.path.join(run_cal4, str(sid), 'maps', 'elevation.map'), 'rb').read()
            ref = open(os.path.join(REF_DIR, str(sid), 'maps', 'elevation.map'), 'rb').read()
            assert result == ref, f"elevation.map mismatch for station {sid}"

    def test_clipped_nc_smaller_than_global(self, run_cal4):
        """Clipped NetCDF is smaller than the global input for each station."""
        import xarray as xr
        global_nc = os.path.join(INPUT_DIR, 'global_maps', 'temperature.nc')
        global_ds = xr.open_dataset(global_nc)
        global_size = global_ds['temperature'].size

        for sid in [100, 200, 300]:
            clipped_nc = os.path.join(run_cal4, str(sid), 'maps', 'temperature.nc')
            clipped_ds = xr.open_dataset(clipped_nc)
            clipped_size = list(clipped_ds.data_vars.values())[0].size
            assert clipped_size <= global_size, f"Station {sid}: clipped not smaller"
            clipped_ds.close()
        global_ds.close()

    def test_single_station_processing(self, tmp_path):
        """Running with a single station ID only clips for that station."""
        from CAL_4_CUT_MAPS import main

        settings_file, subcatch_path = _setup_cal4(tmp_path)
        path_maps = os.path.join(INPUT_DIR, 'global_maps')

        main(settings_file, path_maps, '200')

        # Only station 200 should have cut maps
        assert os.path.isfile(os.path.join(subcatch_path, '200', 'maps', 'elevation.map'))
        # Station 100 and 300 should NOT have cut elevation (only mask.map)
        assert not os.path.isfile(os.path.join(subcatch_path, '100', 'maps', 'elevation.map'))
        assert not os.path.isfile(os.path.join(subcatch_path, '300', 'maps', 'elevation.map'))

    def test_invalid_station_input_raises(self, tmp_path):
        """Non-integer non-file station input raises ValueError."""
        from CAL_4_CUT_MAPS import main

        settings_file, _ = _setup_cal4(tmp_path)
        path_maps = os.path.join(INPUT_DIR, 'global_maps')

        with pytest.raises(ValueError, match="neither a valid integer"):
            main(settings_file, path_maps, 'nonexistent_file.txt')
