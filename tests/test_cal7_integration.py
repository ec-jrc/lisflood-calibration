"""
Integration test for CAL_7_LONGTERM_RUN.

Runs the full long-term run pipeline on catchment 7838 using
pareto_front.csv produced by CAL_6.

Test data lives in tests/data/CAL_5c_6_7/.
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
TABLES_DIR = os.path.join(INPUT_DIR, 'catchments', '7838', 'tables')


def _create_settings_with_pareto(tmp_path):
    """Create resolved settings with pareto_front.csv already in place (from CAL_6)."""
    tmp_catchments = str(tmp_path / 'catchments')
    shutil.copytree(os.path.join(INPUT_DIR, 'catchments'), tmp_catchments)

    # Copy pareto_front.csv from CAL_6 reference into the catchment dir
    shutil.copy2(
        os.path.join(REF_DIR, 'pareto_front_cal6.csv'),
        os.path.join(tmp_catchments, '7838', 'pareto_front.csv'),
    )

    # Resolve XML template
    xml_src = os.path.join(INPUT_DIR, 'templates', 'OSLisfloodGloFASv5calibration_v1.xml')
    xml_dst = str(tmp_path / 'template.xml')
    xml_content = open(xml_src).read()
    xml_content = xml_content.replace('{TABLES_DIR}/', TABLES_DIR + '/')
    with open(xml_dst, 'w') as f:
        f.write(xml_content)

    # Resolve settings file
    template = open(os.path.join(INPUT_DIR, 'settings.txt')).read()
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


class TestCAL7LongtermRun:
    """Integration test for CAL_7 main() on real data.

    Runs the long-term simulation once and checks all expected outputs.
    """

    def test_longterm_run_produces_expected_outputs(self, tmp_path):
        """Run CAL_7 main() and verify all output files and their content."""
        from CAL_7_LONGTERM_RUN import main

        settings_file, tmp_catchments = _create_settings_with_pareto(tmp_path)
        main(settings_file, '7838')

        out_dir = os.path.join(tmp_catchments, '7838', 'out')

        # Check output files exist
        assert os.path.isfile(os.path.join(out_dir, 'streamflow_simulated_best.csv')), \
            "Missing streamflow_simulated_best.csv"
        assert os.path.isfile(os.path.join(out_dir, 'streamflow_simulated_best.tss')), \
            "Missing streamflow_simulated_best.tss"
        assert os.path.isfile(os.path.join(out_dir, 'chanqavgdt_simulated_best.csv')), \
            "Missing chanqavgdt_simulated_best.csv"
        assert os.path.isfile(os.path.join(out_dir, 'chanqavgdt_simulated_best.tss')), \
            "Missing chanqavgdt_simulated_best.tss"

        # Check streamflow CSV content
        df = pd.read_csv(os.path.join(out_dir, 'streamflow_simulated_best.csv'), index_col=0)
        assert len(df) > 0, "streamflow_simulated_best.csv is empty"
        assert '7838' in df.columns, "Station 7838 column missing from streamflow CSV"
        assert df['7838'].notna().sum() > 0, "streamflow_simulated_best.csv has no valid data"

        # Check chanqavgdt CSV content
        df_chanq = pd.read_csv(os.path.join(out_dir, 'chanqavgdt_simulated_best.csv'), index_col=0)
        assert len(df_chanq) > 0, "chanqavgdt_simulated_best.csv is empty"
        assert '7838' in df_chanq.columns, "Station 7838 column missing from chanqavgdt CSV"

        # Compare streamflow against reference (tolerance for floating point)
        ref_df = pd.read_csv(os.path.join(REF_DIR, 'streamflow_simulated_best_cal7.csv'), index_col=0)
        pd.testing.assert_frame_equal(df, ref_df, atol=1e-4, rtol=1e-4)
