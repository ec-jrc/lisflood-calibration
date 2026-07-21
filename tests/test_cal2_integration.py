"""
Integration test for CAL_2_HYDRO_DEPENDENCIES.

Runs the full script on a small synthetic 10x10 LDD map with 3 stations
and verifies the output (stations_links.csv, gauges.map, etc.) against
reference files.

Test data lives in tests/data/CAL_2/ and is self-contained:
- input/ldd.map             10x10 synthetic LDD (356 bytes)
- input/stations.csv        3 stations along a west-flowing river
- reference/stations_links.csv  Expected connectivity output
- reference/gauges.map      Expected station locations map
- reference/inlets.map      Expected inlets map
- reference/interstation_regions.map
- reference/sampling_frequency.map

The station hierarchy is:
  Station 100 (outlet) -> contains 200 -> contains 300
  Direct connections: 100->200, 200->300
"""

import os
import sys
import shutil
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
CAL2_DATA = os.path.join(TEST_DIR, 'data', 'CAL_2')
INPUT_DIR = os.path.join(CAL2_DATA, 'input')
REF_DIR = os.path.join(CAL2_DATA, 'reference')


class TestCAL2HydroDependencies:
    """Integration tests for CAL_2_HYDRO_DEPENDENCIES main()."""

    @pytest.fixture
    def run_cal2(self, tmp_path):
        """Run CAL_2 main() and return the output directory."""
        from CAL_2_HYDRO_DEPENDENCIES import main

        path_result = str(tmp_path / 'output')
        path_temp = str(tmp_path / 'temp')

        main(
            os.path.join(INPUT_DIR, 'stations.csv'),
            os.path.join(INPUT_DIR, 'ldd.map'),
            path_result,
            path_temp,
        )
        return path_result

    def test_stations_links_csv_matches_reference(self, run_cal2):
        """stations_links.csv connectivity matches the reference."""
        result_csv = os.path.join(run_cal2, 'stations_links.csv')
        ref_csv = os.path.join(REF_DIR, 'stations_links.csv')

        result = open(result_csv).read()
        expected = open(ref_csv).read()
        assert result == expected

    def test_station_100_connects_to_200(self, run_cal2):
        """Station 100 has station 200 as direct downstream connection."""
        csv_path = os.path.join(run_cal2, 'stations_links.csv')
        df = pd.read_csv(csv_path, index_col=0)
        row = df.loc[100].dropna().tolist()
        assert 200 in [int(v) for v in row]

    def test_station_200_connects_to_300(self, run_cal2):
        """Station 200 has station 300 as direct downstream connection."""
        csv_path = os.path.join(run_cal2, 'stations_links.csv')
        df = pd.read_csv(csv_path, index_col=0)
        row = df.loc[200].dropna().tolist()
        assert 300 in [int(v) for v in row]

    def test_station_300_is_leaf(self, run_cal2):
        """Station 300 has no direct connections (leaf station)."""
        csv_path = os.path.join(run_cal2, 'stations_links.csv')
        df = pd.read_csv(csv_path, index_col=0)
        row = df.loc[300].dropna().tolist()
        assert row == []

    def test_output_maps_created(self, run_cal2):
        """All expected output map files are created."""
        expected_files = [
            'gauges.map',
            'inlets.map',
            'interstation_regions.map',
            'sampling_frequency.map',
            'stations_links.csv',
        ]
        for fname in expected_files:
            assert os.path.isfile(os.path.join(run_cal2, fname)), f"Missing: {fname}"

    def test_gauges_map_matches_reference(self, run_cal2):
        """gauges.map binary content matches the reference."""
        result = open(os.path.join(run_cal2, 'gauges.map'), 'rb').read()
        reference = open(os.path.join(REF_DIR, 'gauges.map'), 'rb').read()
        assert result == reference

    def test_station_id_too_high_raises(self, tmp_path):
        """Station ID >= 999999 raises an exception."""
        from CAL_2_HYDRO_DEPENDENCIES import main

        # Create a stations CSV with an invalid ID
        stations_csv = tmp_path / 'bad_stations.csv'
        stations_csv.write_text("ObsID,LisfloodX,LisfloodY\n999999,140.1,-20.2\n")

        with pytest.raises(Exception, match="too high"):
            main(
                str(stations_csv),
                os.path.join(INPUT_DIR, 'ldd.map'),
                str(tmp_path / 'out'),
                str(tmp_path / 'tmp'),
            )
