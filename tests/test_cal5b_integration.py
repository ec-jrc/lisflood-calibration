"""
Integration test for CAL_5b_SET_PRERUN_IN_SETTINGS.

Verifies that main() correctly reads the prerun_start date from
the station CSV and writes it into the settings file.
"""

import os
import sys
import pytest
from configparser import ConfigParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))


class TestCAL5bSetPrerunInSettings:
    """Integration tests for CAL_5b_SET_PRERUN_IN_SETTINGS main()."""

    def _create_inputs(self, tmp_path, prerun_values=None):
        """Create settings file and stations CSV for testing."""
        if prerun_values is None:
            prerun_values = {100: '02/01/1979 00:00', 200: '01/01/2000 00:00'}

        # Create stations CSV with prerun_start column
        stations_csv = str(tmp_path / 'stations.csv')
        lines = ['ObsID,LisfloodX,LisfloodY,prerun_start']
        for obsid, prerun in prerun_values.items():
            lines.append(f'{obsid},140.0,-20.0,{prerun}')
        with open(stations_csv, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        # Create settings file with an initial prerun_start value
        settings_file = str(tmp_path / 'settings.txt')
        with open(settings_file, 'w') as f:
            f.write("[Main]\n")
            f.write("prerun_start = 01/01/1990 00:00\n")
            f.write("forcing_start = 02/01/2010 00:00\n")
            f.write("forcing_end = 31/12/2023 00:00\n")

        return settings_file, stations_csv

    def test_prerun_start_updated_for_station(self, tmp_path):
        """prerun_start in settings is updated to the value from station CSV."""
        from CAL_5b_SET_PRERUN_IN_SETTINGS import main

        settings_file, stations_csv = self._create_inputs(tmp_path)
        main(settings_file, stations_csv, '100')

        parser = ConfigParser()
        parser.read(settings_file)
        assert parser.get('Main', 'prerun_start') == '02/01/1979 00:00'

    def test_different_station_different_date(self, tmp_path):
        """Each station can have a different prerun_start."""
        from CAL_5b_SET_PRERUN_IN_SETTINGS import main

        settings_file, stations_csv = self._create_inputs(tmp_path)
        main(settings_file, stations_csv, '200')

        parser = ConfigParser()
        parser.read(settings_file)
        assert parser.get('Main', 'prerun_start') == '01/01/2000 00:00'

    def test_other_settings_preserved(self, tmp_path):
        """Other settings in the file are not lost after update."""
        from CAL_5b_SET_PRERUN_IN_SETTINGS import main

        settings_file, stations_csv = self._create_inputs(tmp_path)
        main(settings_file, stations_csv, '100')

        parser = ConfigParser()
        parser.read(settings_file)
        assert parser.get('Main', 'forcing_start') == '02/01/2010 00:00'
        assert parser.get('Main', 'forcing_end') == '31/12/2023 00:00'

    def test_missing_station_raises(self, tmp_path):
        """Looking up a non-existent station raises an Exception."""
        from CAL_5b_SET_PRERUN_IN_SETTINGS import main

        settings_file, stations_csv = self._create_inputs(tmp_path)
        with pytest.raises(Exception, match="not found"):
            main(settings_file, stations_csv, '9999')

    def test_missing_settings_file_raises(self, tmp_path):
        """A non-existent settings file raises FileNotFoundError."""
        from CAL_5b_SET_PRERUN_IN_SETTINGS import main

        _, stations_csv = self._create_inputs(tmp_path)
        with pytest.raises(FileNotFoundError):
            main(str(tmp_path / 'nonexistent.txt'), stations_csv, '100')
