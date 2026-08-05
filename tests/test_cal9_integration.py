"""
Integration test for CAL_9_DIAGNOSTICS.

Calls main() with synthetic station data and verifies that
discharge and variable HTML plots are created with correct content.

Requires plotly.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

plotly = pytest.importorskip("plotly", reason="plotly required for CAL_9 tests")


def _create_diagnostic_data(tmp_path, catchment_id=999):
    """Create the full directory structure expected by CAL_9's construct_dfs."""
    from liscal.diagnostic_plots import plot_groupings

    base = tmp_path / 'catchments'
    catch_dir = base / str(catchment_id)
    station_dir = catch_dir / 'station'
    out_dir = catch_dir / 'out'
    longterm_dir = out_dir / 'long_term_run'
    station_dir.mkdir(parents=True)
    longterm_dir.mkdir(parents=True)

    n_steps = 100
    dates = pd.date_range('01/01/2017', periods=n_steps, freq='D')

    # observations.csv
    np.random.seed(42)
    obs_values = np.random.uniform(5, 50, n_steps)
    obs_df = pd.DataFrame({
        'time': dates.strftime('%d/%m/%Y %H:%M'),
        'dis': obs_values
    })
    obs_df.to_csv(station_dir / 'observations.csv', index=False)

    # station_data.csv
    station_data = pd.DataFrame({
        str(catchment_id): ['TestStation', 'XX']
    }, index=['StationName', 'Country code'])
    station_data.to_csv(station_dir / 'station_data.csv')

    # pHistoryWRanks.csv
    phistory = pd.DataFrame({'Kling Gupta Efficiency': [0.75]}, index=[0])
    phistory.to_csv(catch_dir / 'pHistoryWRanks.csv')

    # streamflow_simulated_best.csv
    sim_values = obs_values * 0.9 + np.random.uniform(-1, 1, n_steps)
    sim_df = pd.DataFrame({
        'time': dates.strftime('%d/%m/%Y %H:%M'),
        'dis': sim_values
    })
    sim_df.to_csv(out_dir / 'streamflow_simulated_best.csv', index=False)

    # Create .tss files for each variable in plot_groupings
    np.random.seed(123)
    all_vars = [var for group in plot_groupings for subplot in group for var in subplot]
    for var in all_vars:
        tss_content = "timeseries scalar\n2\ntimestep\n1\n"
        for i in range(n_steps):
            tss_content += f"{i+1} {np.random.uniform(0, 10):.4f}\n"
        (longterm_dir / f'{var}.tss').write_text(tss_content)

    # Settings file
    settings_file = tmp_path / 'settings.txt'
    settings_file.write_text(f"[Path]\nsubcatchment_path = {base}\n")

    # Catchments list
    catchments_file = tmp_path / 'catchments.csv'
    catchments_file.write_text(f"{catchment_id}\n")

    # savepath is used as prefix: f"{savepath}{catchment_id}_dis.html"
    savepath = str(tmp_path / 'plots') + '/'

    return str(settings_file), str(catchments_file), savepath


class TestCAL9Diagnostics:
    """Integration test calling CAL_9 main() with synthetic data."""

    def test_main_generates_html_plots(self, tmp_path):
        """Call main() and verify HTML discharge + variable group plots are created."""
        from CAL_9_DIAGNOSTICS import main

        settings_file, catchments_file, savepath = _create_diagnostic_data(tmp_path)

        # Create output directory (savepath is used as prefix ending with '/')
        os.makedirs(savepath.rstrip('/'), exist_ok=True)

        main(settings_file, catchments_file, savepath)

        # Discharge plot
        dis_plot = f"{savepath}999_dis.html"
        assert os.path.isfile(dis_plot), "main() did not produce discharge HTML plot"
        assert os.path.getsize(dis_plot) > 1000, "Discharge plot is too small"

        # Variable group plots (one per plot_groupings entry = 4)
        plots_dir = savepath.rstrip('/')
        assert os.path.isdir(plots_dir), f"plots directory not found: {plots_dir}"
        html_files = [f for f in os.listdir(plots_dir) if f.endswith('.html')]
        # 1 discharge + 4 variable group plots = 5
        assert len(html_files) == 5, f"Expected 5 HTML plots, got {len(html_files)}: {html_files}"

        # Verify discharge plot contains the station name
        dis_content = open(dis_plot).read()
        assert 'TestStation' in dis_content, "Discharge plot missing station name"
        assert 'model' in dis_content, "Discharge plot missing 'model' trace"
        assert 'obs' in dis_content, "Discharge plot missing 'obs' trace"
