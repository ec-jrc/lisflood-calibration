"""
Integration test for CAL_6_CALIBRATION.

Runs the full calibration pipeline on catchment 7838 with fast_debug=1
(2 generations, 2 individuals, 120-day run) using 2 years of forcing data.

Test data lives in tests/data/CAL_5c_6_7/.
"""

import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data', 'CAL_5c_6_7')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
REF_DIR = os.path.join(DATA_DIR, 'reference')
TABLES_DIR = os.path.join(INPUT_DIR, 'catchments', '7838', 'tables')


def _create_settings(tmp_path):
    """Create resolved settings and XML template for a fresh CAL_6 run."""
    tmp_catchments = str(tmp_path / 'catchments')
    shutil.copytree(os.path.join(INPUT_DIR, 'catchments'), tmp_catchments)

    # Resolve XML template: replace {TABLES_DIR} placeholder
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


class TestCAL6Calibration:
    """Integration test for CAL_6 main() on real data.

    Runs the calibration once and checks all expected outputs.
    """

    def test_calibration_produces_expected_outputs(self, tmp_path):
        """Run CAL_6 main() and verify pareto_front.csv against reference."""
        from CAL_6_CALIBRATION import main

        settings_file, tmp_catchments = _create_settings(tmp_path)
        main(settings_file, '7838', n_cpus=1, seed=13)

        catchment_dir = os.path.join(tmp_catchments, '7838')

        # Check pareto_front.csv exists
        pareto_path = os.path.join(catchment_dir, 'pareto_front.csv')
        assert os.path.isfile(pareto_path), "Missing pareto_front.csv"

        # Check output run directories created
        out_dir = os.path.join(catchment_dir, 'out')
        assert os.path.isdir(out_dir), "Missing out/ directory"
        assert len(os.listdir(out_dir)) > 0, "out/ directory is empty"

        # Load result and reference
        result = pd.read_csv(pareto_path)
        ref = pd.read_csv(os.path.join(REF_DIR, 'pareto_front_cal6.csv'))

        # Check structure matches
        assert list(result.columns) == list(ref.columns), \
            f"Column mismatch.\n  Result: {list(result.columns)}\n  Reference: {list(ref.columns)}"

        # Compare parameter values with tolerance
        param_cols = [c for c in result.columns if c.startswith('param_')]
        np.testing.assert_allclose(
            result[param_cols].values,
            ref[param_cols].values,
            rtol=1e-4, atol=1e-4,
            err_msg="Calibrated parameters differ from reference",
        )
