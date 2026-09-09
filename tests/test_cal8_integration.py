"""
Integration test for CAL_8_POSTPROCESSING.

Runs main() on catchment 7838 using the streamflow_simulated_best.csv
from CAL_7 reference and verifies statistics against reference values.

Test data lives in tests/data/CAL_5c_6_7/.
"""

import os
import sys
import shutil
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data', 'CAL_5c_6_7')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
REF_DIR = os.path.join(DATA_DIR, 'reference')
TABLES_DIR = os.path.join(INPUT_DIR, 'catchments', '7838', 'tables')


def _create_settings_for_postprocessing(tmp_path):
    """Create resolved settings for CAL_8 with streamflow_simulated_best.csv in place."""
    tmp_catchments = str(tmp_path / 'catchments')
    shutil.copytree(os.path.join(INPUT_DIR, 'catchments'), tmp_catchments)

    out_dir = os.path.join(tmp_catchments, '7838', 'out')
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(REF_DIR, 'streamflow_simulated_best_cal7.csv'),
        os.path.join(out_dir, 'streamflow_simulated_best.csv'),
    )

    shutil.copy2(
        os.path.join(REF_DIR, 'pareto_front_cal6.csv'),
        os.path.join(tmp_catchments, '7838', 'pareto_front.csv'),
    )

    # Patch station_data.csv: Obs_start must match forcing_start (02/01/2017)
    station_data_file = os.path.join(tmp_catchments, '7838', 'station', 'station_data.csv')
    station_data = pd.read_csv(station_data_file, index_col=0)
    station_data.loc['Obs_start', '7838'] = '02/01/2017 00:00'
    station_data.to_csv(station_data_file)

    # Trim observations.csv to start at 02/01/2017 (matching forcing_start)
    obs_file = os.path.join(tmp_catchments, '7838', 'station', 'observations.csv')
    obs_df = pd.read_csv(obs_file, index_col=0)
    obs_df = obs_df.iloc[1:]
    obs_df.to_csv(obs_file)

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
    summary_path = str(tmp_path / 'summary')
    content = content.replace('[Path]\n', f'[Path]\nsummary_path = {summary_path}\n')

    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)

    return settings_file, tmp_catchments


class TestCAL8Postprocessing:
    """Integration test calling CAL_8 main() on real data (station 7838)."""

    def test_main_produces_correct_stats(self, tmp_path, monkeypatch):
        """Call main() and verify stats.txt matches reference KGE/NSE values."""
        # Disable LaTeX (not installed in test env)
        from liscal.config import PlotParameters
        monkeypatch.setattr(PlotParameters, 'text', {
            'figure': {'autolayout': True},
            'font': {'size': 14, 'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'], 'weight': 'bold'},
            'text': {'usetex': False},
            'axes': {'labelweight': 'bold'},
        })
        # Mock SpatialPlot.plot (needs pcraster map files not in test data)
        import matplotlib.pyplot as plt
        from liscal import evaluation
        monkeypatch.setattr(evaluation.SpatialPlot, 'plot', lambda self, path_out, subcatch_dir: plt.figure())

        from CAL_8_POSTPROCESSING import main

        settings_file, tmp_catchments = _create_settings_for_postprocessing(tmp_path)
        main(settings_file, '7838')

        out_dir = os.path.join(tmp_catchments, '7838', 'out')

        # Verify stats.txt was created and contains reference KGE values
        stats_file = os.path.join(out_dir, 'stats.txt')
        assert os.path.isfile(stats_file), "main() did not produce stats.txt"
        stats_content = open(stats_file).read()
        # Parse the stats dict from the text file
        stats = eval(stats_content)

        # Compare against reference values (from CAL_7 streamflow vs observations)
        assert stats['kge'] == pytest.approx(-0.05423, abs=1e-4)
        assert stats['corr'] == pytest.approx(0.46898, abs=1e-4)
        assert stats['bias'] == pytest.approx(1.11272, abs=1e-4)
        assert stats['spread'] == pytest.approx(0.09628, abs=1e-4)
        assert stats['nse'] == pytest.approx(0.08862, abs=1e-4)

        # Verify plot SVG/PNG files were created
        for plot_name in ['speedo.svg', 'boxy.svg', 'timmy.svg', 'qqy.svg']:
            assert os.path.isfile(os.path.join(out_dir, plot_name)), f"Missing {plot_name}"
            assert os.path.getsize(os.path.join(out_dir, plot_name)) > 100

        for plot_name in ['bestparmtrs.png']:
            assert os.path.isfile(os.path.join(out_dir, plot_name)), f"Missing {plot_name}"
