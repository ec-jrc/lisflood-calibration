"""
Integration test for CAL_5c_FORCING_STATS.

Runs the full script on real catchment 7838 data (2 years of forcing,
2017-2018) and verifies that the Budyko and temperature statistics
are correctly computed and written to station_data.csv.

Test data lives in tests/data/CAL_5c_6_7/:
- input/settings.txt                  Settings template
- input/catchments/7838/maps/         Static maps + 2 years of forcing
- input/catchments/7838/station/      Station observations and metadata
- input/catchments/7838/inflow/       Empty inflow map (leaf station)
- input/templates/                    param_ranges + LISFLOOD XML template
- input/stations/                     Global station metadata + observations
- reference/station_data_after_cal5c.csv  Expected output
"""

import os
import sys
import shutil
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data', 'CAL_5c_6_7')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
REF_DIR = os.path.join(DATA_DIR, 'reference')


def _create_settings(tmp_path):
    """Create resolved settings file with paths pointing to test data."""
    template = open(os.path.join(INPUT_DIR, 'settings.txt')).read()
    # Use a tmp copy of catchments so we don't modify the input in-place
    tmp_catchments = str(tmp_path / 'catchments')
    shutil.copytree(os.path.join(INPUT_DIR, 'catchments'), tmp_catchments)

    # Resolve XML template: replace {TABLES_DIR} placeholder
    tables_dir = os.path.join(tmp_catchments, '7838', 'tables')
    xml_src = os.path.join(INPUT_DIR, 'templates', 'OSLisfloodGloFASv5calibration_v1.xml')
    xml_dst = str(tmp_path / 'template.xml')
    xml_content = open(xml_src).read()
    xml_content = xml_content.replace('{TABLES_DIR}/', tables_dir + '/')
    with open(xml_dst, 'w') as f:
        f.write(xml_content)

    content = template.replace('{INPUT_DIR}', INPUT_DIR)
    content = content.replace(
        f'subcatchment_path = {INPUT_DIR}/catchments',
        f'subcatchment_path = {tmp_catchments}',
    )
    content = content.replace(
        f'LISFLOODSettings = {INPUT_DIR}/templates/OSLisfloodGloFASv5calibration_v1.xml',
        f'LISFLOODSettings = {xml_dst}',
    )
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)
    return settings_file, tmp_catchments


class TestCAL5cForcingStats:
    """Integration tests for CAL_5c_FORCING_STATS main()."""

    @pytest.fixture
    def run_cal5c(self, tmp_path):
        """Run CAL_5c for station 7838 and return the station output dir."""
        from CAL_5c_FORCING_STATS import main

        settings_file, tmp_catchments = _create_settings(tmp_path)
        main(settings_file, '7838')
        return os.path.join(tmp_catchments, '7838', 'station')

    def test_budyko_fields_added(self, run_cal5c):
        """CAL_5c adds precip_budyko, PET_budyko, min_TAvgS, min_AridIdx."""
        df = pd.read_csv(os.path.join(run_cal5c, 'station_data.csv'), index_col=0)
        for field in ['precip_budyko', 'PET_budyko', 'min_TAvgS', 'min_AridIdx']:
            assert field in df.index, f"Missing field: {field}"

    def test_precip_budyko_matches_reference(self, run_cal5c):
        """precip_budyko value matches the reference."""
        result = pd.read_csv(os.path.join(run_cal5c, 'station_data.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(REF_DIR, 'station_data_after_cal5c.csv'), index_col=0)

        result_val = float(result.loc['precip_budyko'].values[0])
        ref_val = float(ref.loc['precip_budyko'].values[0])
        assert result_val == pytest.approx(ref_val, rel=1e-6)

    def test_pet_budyko_matches_reference(self, run_cal5c):
        """PET_budyko value matches the reference."""
        result = pd.read_csv(os.path.join(run_cal5c, 'station_data.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(REF_DIR, 'station_data_after_cal5c.csv'), index_col=0)

        result_val = float(result.loc['PET_budyko'].values[0])
        ref_val = float(ref.loc['PET_budyko'].values[0])
        assert result_val == pytest.approx(ref_val, rel=1e-6)

    def test_min_tavgs_matches_reference(self, run_cal5c):
        """min_TAvgS value matches the reference."""
        result = pd.read_csv(os.path.join(run_cal5c, 'station_data.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(REF_DIR, 'station_data_after_cal5c.csv'), index_col=0)

        result_val = float(result.loc['min_TAvgS'].values[0])
        ref_val = float(ref.loc['min_TAvgS'].values[0])
        assert result_val == pytest.approx(ref_val, rel=1e-6)

    def test_min_aridx_matches_reference(self, run_cal5c):
        """min_AridIdx value matches the reference."""
        result = pd.read_csv(os.path.join(run_cal5c, 'station_data.csv'), index_col=0)
        ref = pd.read_csv(os.path.join(REF_DIR, 'station_data_after_cal5c.csv'), index_col=0)

        result_val = float(result.loc['min_AridIdx'].values[0])
        ref_val = float(ref.loc['min_AridIdx'].values[0])
        assert result_val == pytest.approx(ref_val, rel=1e-6)

    def test_aridity_index_below_half(self, run_cal5c):
        """For this arid catchment, aridity index < 0.5 (TransLoss should be calibrated)."""
        df = pd.read_csv(os.path.join(run_cal5c, 'station_data.csv'), index_col=0)
        aridity = float(df.loc['min_AridIdx'].values[0])
        assert aridity < 0.5
