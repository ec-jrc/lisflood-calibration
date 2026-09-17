"""
Integration test for CAL_11_COMBINE_RESULTS.

Calls main() with a synthetic calibration directory structure
and verifies the summary CSV output matches expected values.

The spatial part (get_mask) requires pcraster+real maps and is mocked.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))


def _create_cal11_test_data(tmp_path):
    """Create a complete directory structure for CAL_11 main()."""
    main_dir = tmp_path / 'project'
    data_dir = main_dir / 'data' / 'stations'
    cal_dir = main_dir / 'calibration'
    data_dir.mkdir(parents=True)
    cal_dir.mkdir(parents=True)

    # stations_data.csv (3 stations, one without results)
    stations = pd.DataFrame({
        'ObsID': [100, 200, 300],
        'StationName': ['Danube_Outlet', 'Rhine_Basel', 'Elbe_Dresden'],
        'River': ['Danube', 'Rhine', 'Elbe'],
        'StationLon': [25.0, 7.5, 13.7],
        'StationLat': [44.4, 47.5, 51.0],
        'DrainingArea.km2.LDD': [800000, 36000, 53000],
    })
    stations.to_csv(data_dir / 'stations_data.csv', index=False)

    # pHistoryWRanks.csv for stations 100 and 200 (not 300)
    for obsid, kge, corr in [(100, 0.82, 0.91), (200, 0.71, 0.85)]:
        catch_dir = cal_dir / str(obsid)
        catch_dir.mkdir()
        phistory = pd.DataFrame({
            'randId': ['abc123'],
            'UpperZoneTimeConstant': [15.5],
            'LowerZoneTimeConstant': [200.0],
            'Kling Gupta Efficiency': [kge],
            'Correlation': [corr],
        })
        phistory.to_csv(catch_dir / 'pHistoryWRanks.csv', index=True)

    # settings file path: main() derives main_dir by stripping '/calibration/settings.txt'
    settings_file = str(cal_dir / 'settings.txt')
    open(settings_file, 'w').close()

    return settings_file, str(main_dir)


class TestCAL11CombineResults:
    """Integration test calling CAL_11 main()."""

    def test_main_produces_correct_summary_csv(self, tmp_path):
        """Call main() and verify calibration_summary.csv matches expected merge."""
        import CAL_11_COMBINE_RESULTS as cal11

        settings_file, main_dir = _create_cal11_test_data(tmp_path)

        # Mock get_mask (needs pcraster + real spatial maps)
        with patch.object(cal11, 'get_mask', side_effect=Exception("no spatial data")):
            cal11.main(settings_file, '/nonexistent/cutmaps')

        summary_csv = os.path.join(main_dir, 'calibration', 'summary', 'calibration_summary.csv')
        assert os.path.isfile(summary_csv), "main() did not produce calibration_summary.csv"

        df = pd.read_csv(summary_csv)

        # All 3 stations present (outer merge)
        assert len(df) == 3

        # Verify reference values for calibrated stations
        row_100 = df[df['ObsID'] == 100].iloc[0]
        assert row_100['Kling Gupta Efficiency'] == pytest.approx(0.82)
        assert row_100['Correlation'] == pytest.approx(0.91)
        assert row_100['UpperZoneTimeConstant'] == pytest.approx(15.5)

        row_200 = df[df['ObsID'] == 200].iloc[0]
        assert row_200['Kling Gupta Efficiency'] == pytest.approx(0.71)
        assert row_200['LowerZoneTimeConstant'] == pytest.approx(200.0)

        # Station 300 (no pHistoryWRanks) has NaN calibration fields
        row_300 = df[df['ObsID'] == 300].iloc[0]
        assert pd.isna(row_300['Kling Gupta Efficiency'])

        # Metadata columns preserved
        assert row_100['StationName'] == 'Danube_Outlet'
        assert row_200['River'] == 'Rhine'
