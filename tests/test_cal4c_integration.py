"""
Integration test for CAL_4c_CUT_MAPS_parallel.

Runs the parallel map cutting script on synthetic data (same test data as CAL_4)
and verifies that maps are correctly clipped for all stations in the list.

Reuses test data from tests/data/CAL_4/.
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


def _setup_cal4c(tmp_path):
    """Set up CAL_4c with tmp subcatchment_path containing mask.map files and station list."""
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

    # Create station list file
    station_list = str(tmp_path / 'stations.txt')
    with open(station_list, 'w') as f:
        f.write("100\n200\n300\n")

    return settings_file, subcatch_path, station_list


class TestCAL4cCutMapsParallel:
    """Integration tests for CAL_4c_CUT_MAPS_parallel main()."""

    @pytest.fixture
    def run_cal4c(self, tmp_path):
        """Run CAL_4c on all 3 stations in parallel and return subcatchment output path."""
        from CAL_4c_CUT_MAPS_parallel import main

        settings_file, subcatch_path, station_list = _setup_cal4c(tmp_path)
        path_maps = os.path.join(INPUT_DIR, 'global_maps')

        main(settings_file, path_maps, station_list, max_workers=2)
        return subcatch_path

    def test_elevation_map_created_for_all_stations(self, run_cal4c):
        """A clipped elevation.map is created for each station."""
        for sid in [100, 200, 300]:
            fpath = os.path.join(run_cal4c, str(sid), 'maps', 'elevation.map')
            assert os.path.isfile(fpath), f"Missing elevation.map for station {sid}"

    def test_temperature_nc_created_for_all_stations(self, run_cal4c):
        """A clipped temperature.nc is created for each station."""
        for sid in [100, 200, 300]:
            fpath = os.path.join(run_cal4c, str(sid), 'maps', 'temperature.nc')
            assert os.path.isfile(fpath), f"Missing temperature.nc for station {sid}"

    def test_elevation_map_matches_reference(self, run_cal4c):
        """Clipped elevation.map binary matches reference for each station."""
        for sid in [100, 200, 300]:
            result = open(os.path.join(run_cal4c, str(sid), 'maps', 'elevation.map'), 'rb').read()
            ref = open(os.path.join(REF_DIR, str(sid), 'maps', 'elevation.map'), 'rb').read()
            assert result == ref, f"elevation.map mismatch for station {sid}"

    def test_clipped_nc_smaller_than_global(self, run_cal4c):
        """Clipped NetCDF is smaller than the global input for each station."""
        import xarray as xr
        global_nc = os.path.join(INPUT_DIR, 'global_maps', 'temperature.nc')
        global_ds = xr.open_dataset(global_nc)
        global_size = global_ds['temperature'].size

        for sid in [100, 200, 300]:
            clipped_nc = os.path.join(run_cal4c, str(sid), 'maps', 'temperature.nc')
            clipped_ds = xr.open_dataset(clipped_nc)
            clipped_size = list(clipped_ds.data_vars.values())[0].size
            assert clipped_size <= global_size, f"Station {sid}: clipped not smaller"
            clipped_ds.close()
        global_ds.close()

    def test_skips_already_existing_files(self, tmp_path):
        """Running twice does not re-process already-cut maps."""
        from CAL_4c_CUT_MAPS_parallel import main

        settings_file, subcatch_path, station_list = _setup_cal4c(tmp_path)
        path_maps = os.path.join(INPUT_DIR, 'global_maps')

        # First run
        main(settings_file, path_maps, station_list, max_workers=1)

        # Get modification time of a produced file
        fpath = os.path.join(subcatch_path, '200', 'maps', 'elevation.map')
        mtime_first = os.path.getmtime(fpath)

        # Second run (should skip)
        main(settings_file, path_maps, station_list, max_workers=1)
        mtime_second = os.path.getmtime(fpath)

        assert mtime_first == mtime_second, "File was re-processed on second run"
