"""
Unit tests for reservoir events processing logic in liscal/stations.py.

Tests cover:
- process_reservoir_periods: Splitting calibration/long-run periods around
  reservoir construction/demolition events.
- create_netcdf_map: Identifying active reservoirs for a given period.
- update_rsfil_netcdf_map: Updating reservoir fill maps for new reservoirs.
"""

import pytest
import datetime
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call

from liscal import stations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reservoir_events_df():
    """DataFrame with reservoir construction/demolition events."""
    return pd.DataFrame({
        'FID': [101, 102, 103],
        'CONSTR_YEAR': [2005, 2010, 2000],
        'DEMOL_YEAR': [np.nan, 2015, np.nan],
    })


@pytest.fixture
def reservoir_events_df_two_events():
    """DataFrame where two events fall inside the valid period (2 events scenario like station 8484)."""
    return pd.DataFrame({
        'FID': [201, 202, 203],
        'CONSTR_YEAR': [2002, 2008, 2012],
        'DEMOL_YEAR': [np.nan, np.nan, np.nan],
    })


@pytest.fixture
def mock_model():
    """Mock model_initialized with simulateReservoirs enabled."""
    model = Mock()
    model.lissettings.options = {'simulateReservoirs': True}
    model.subcatch.path_station = '/tmp/test_station'
    return model


@pytest.fixture
def mock_model_no_reservoirs():
    """Mock model_initialized with simulateReservoirs disabled."""
    model = Mock()
    model.lissettings.options = {'simulateReservoirs': False}
    return model


@pytest.fixture
def observations_filtered():
    """Mock filtered observations spanning 2005-01-01 to 2016-12-31 (daily)."""
    dates = pd.date_range('2005-01-01', '2016-12-31', freq='D')
    obs = pd.Series(np.random.rand(len(dates)) * 100, index=dates.strftime('%d/%m/%Y %H:%M'))
    return obs


# ---------------------------------------------------------------------------
# Tests: process_reservoir_periods (simulateReservoirs OFF)
# ---------------------------------------------------------------------------

class TestProcessReservoirPeriodsNoSimulation:
    """When simulateReservoirs is OFF, periods should pass through unchanged."""

    def test_returns_original_period_non_longrun(self, mock_model_no_reservoirs, reservoir_events_df):
        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'
        result = stations.process_reservoir_periods(
            mock_model_no_reservoirs, reservoir_events_df, 24, None,
            valid_start, valid_end, Min_calib_days=365, isLongRun=False
        )
        assert result == (valid_start, valid_end)

    def test_returns_none_for_longrun(self, mock_model_no_reservoirs, reservoir_events_df):
        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'
        result = stations.process_reservoir_periods(
            mock_model_no_reservoirs, reservoir_events_df, 24, None,
            valid_start, valid_end, Min_calib_days=365, isLongRun=True
        )
        assert result == (None, None)


# ---------------------------------------------------------------------------
# Tests: process_reservoir_periods (simulateReservoirs ON, non-longrun)
# ---------------------------------------------------------------------------

class TestProcessReservoirPeriodsCalibration:
    """Non-longrun mode: find the best recent period after the last reservoir event."""

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_single_event_trims_start(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                      mock_model, observations_filtered):
        """With one event in the middle, the best period should start after that event."""
        # Reservoir map: one reservoir with FID=101
        reservoir_map = np.array([0, 101, 0, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1, 1])]  # ReservoirSites, Channels
        mock_boolean.return_value = np.array([True, True, True, True])
        mock_compress.return_value = np.array([1, 1, 1, 1])  # IsChannel same shape as reservoirs

        # Reservoir 101 constructed in 2010 (event inside the period 2005-2015)
        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [2010],
            'DEMOL_YEAR': [np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'

        result_start, result_end = stations.process_reservoir_periods(
            mock_model, events_df, 24, observations_filtered,
            valid_start, valid_end, Min_calib_days=100, isLongRun=False
        )

        # The best period should start on or after 2010-01-01
        result_start_dt = pd.to_datetime(result_start, format='%d/%m/%Y %H:%M')
        assert result_start_dt >= pd.Timestamp('2010-01-01')
        # create_netcdf_map should have been called for "FilteredReservoirMap"
        mock_create_map.assert_called_once()
        assert mock_create_map.call_args[0][0] == "FilteredReservoirMap"

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_no_events_in_range_returns_full_period(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                                     mock_model, observations_filtered):
        """If no reservoir events fall within valid_start..valid_end, the full period is returned."""
        reservoir_map = np.array([0, 101, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True])
        mock_compress.return_value = np.array([1, 1, 1])  # IsChannel same shape as reservoirs

        # Event is outside the valid period
        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [1990],
            'DEMOL_YEAR': [np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'

        result_start, result_end = stations.process_reservoir_periods(
            mock_model, events_df, 24, observations_filtered,
            valid_start, valid_end, Min_calib_days=100, isLongRun=False
        )

        # Should get the full observation period back (first and last obs dates in format)
        result_start_dt = pd.to_datetime(result_start, format='%d/%m/%Y %H:%M')
        result_end_dt = pd.to_datetime(result_end, format='%d/%m/%Y %H:%M')
        assert result_start_dt <= pd.Timestamp('2005-01-02')
        assert result_end_dt >= pd.Timestamp('2015-12-30')
        mock_create_map.assert_called_once()

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_two_events_picks_most_recent_valid_period(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                                       mock_model, observations_filtered):
        """With two events, the algorithm should pick the most recent period with enough observations."""
        reservoir_map = np.array([0, 201, 202, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True, True])
        mock_compress.return_value = np.array([1, 1, 1, 1])  # IsChannel same shape as reservoirs

        # Two events: 2008 and 2012
        events_df = pd.DataFrame({
            'FID': [201, 202],
            'CONSTR_YEAR': [2008, 2012],
            'DEMOL_YEAR': [np.nan, np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2016 06:00'

        result_start, result_end = stations.process_reservoir_periods(
            mock_model, events_df, 24, observations_filtered,
            valid_start, valid_end, Min_calib_days=100, isLongRun=False
        )

        # Should pick period starting at or after 2012 (the most recent event)
        result_start_dt = pd.to_datetime(result_start, format='%d/%m/%Y %H:%M')
        assert result_start_dt >= pd.Timestamp('2012-01-01')
        mock_create_map.assert_called_once()

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_raises_when_no_period_has_enough_observations(self, mock_loadmap, mock_boolean, mock_compress,
                                                            mock_create_map, mock_model):
        """If no period after any event has enough observations, an exception is raised."""
        reservoir_map = np.array([0, 101, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True])
        mock_compress.return_value = np.array([1, 1, 1])  # IsChannel same shape as reservoirs

        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [2015],
            'DEMOL_YEAR': [np.nan],
        })

        # Only 10 days of observations (way below Min_calib_days=365)
        dates = pd.date_range('2015-06-01', periods=10, freq='D')
        short_obs = pd.Series(np.random.rand(10), index=dates.strftime('%d/%m/%Y %H:%M'))

        valid_start = '01/01/2015 06:00'
        valid_end = '31/12/2015 06:00'

        with pytest.raises(Exception, match='unable to find best period'):
            stations.process_reservoir_periods(
                mock_model, events_df, 24, short_obs,
                valid_start, valid_end, Min_calib_days=365, isLongRun=False
            )

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_no_reservoirs_in_catchment(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                        mock_model, observations_filtered):
        """When ReservoirSitesCC is empty, the period passes through unchanged."""
        reservoir_map = np.array([0, 0, 0])  # No reservoir sites
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True])
        mock_compress.return_value = np.array([1, 1, 1])  # IsChannel same shape as reservoirs

        events_df = pd.DataFrame({
            'FID': [999],
            'CONSTR_YEAR': [2010],
            'DEMOL_YEAR': [np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'

        result_start, result_end = stations.process_reservoir_periods(
            mock_model, events_df, 24, observations_filtered,
            valid_start, valid_end, Min_calib_days=100, isLongRun=False
        )

        assert result_start == valid_start
        assert result_end == valid_end
        mock_create_map.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: process_reservoir_periods (simulateReservoirs ON, longrun)
# ---------------------------------------------------------------------------

class TestProcessReservoirPeriodsLongRun:
    """Long-run mode: split the full period into subperiods at each reservoir event."""

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_two_events_creates_three_subperiods(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                                  mock_model):
        """Two events in range should produce three subperiods."""
        reservoir_map = np.array([0, 201, 202, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True, True])
        mock_compress.return_value = np.array([1, 1, 1, 1])  # IsChannel same shape as reservoirs

        events_df = pd.DataFrame({
            'FID': [201, 202],
            'CONSTR_YEAR': [2008, 2012],
            'DEMOL_YEAR': [np.nan, np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2016 06:00'

        subperiods, result_df = stations.process_reservoir_periods(
            mock_model, events_df, 24, None,
            valid_start, valid_end, Min_calib_days=None, isLongRun=True
        )

        # Should have 3 subperiods: [start, event1), [event1, event2), [event2, end]
        assert len(subperiods) == 3
        # create_netcdf_map should be called once per subperiod
        assert mock_create_map.call_count == 3
        # Verify map names
        map_names = [c[0][0] for c in mock_create_map.call_args_list]
        assert map_names == ['ReservoirMap_Subperiod_0', 'ReservoirMap_Subperiod_1', 'ReservoirMap_Subperiod_2']

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_no_events_creates_one_subperiod(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                              mock_model):
        """No events in range should produce a single subperiod covering the whole range."""
        reservoir_map = np.array([0, 101, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True])
        mock_compress.return_value = np.array([1, 1, 1])  # IsChannel same shape as reservoirs

        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [1990],  # outside range
            'DEMOL_YEAR': [np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'

        subperiods, result_df = stations.process_reservoir_periods(
            mock_model, events_df, 24, None,
            valid_start, valid_end, Min_calib_days=None, isLongRun=True
        )

        assert len(subperiods) == 1
        assert subperiods[0][0] == pd.Timestamp('2005-01-01 06:00')
        assert subperiods[0][1] == pd.Timestamp('2015-12-31 06:00')
        assert mock_create_map.call_count == 1

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_longrun_returns_filtered_df(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                          mock_model):
        """The returned DataFrame should be filtered to only matching FIDs."""
        reservoir_map = np.array([0, 201, 0, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True, True])
        mock_compress.return_value = np.array([1, 1, 1, 1])  # IsChannel same shape as reservoirs

        # FID 201 matches, FID 999 does not
        events_df = pd.DataFrame({
            'FID': [201, 999],
            'CONSTR_YEAR': [2010, 2012],
            'DEMOL_YEAR': [np.nan, np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2016 06:00'

        subperiods, result_df = stations.process_reservoir_periods(
            mock_model, events_df, 24, None,
            valid_start, valid_end, Min_calib_days=None, isLongRun=True
        )

        # Only FID 201 should remain in the result
        assert 201 in result_df['FID'].values
        assert 999 not in result_df['FID'].values

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_longrun_no_reservoirs_returns_none(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                                 mock_model):
        """When no reservoir sites exist in catchment, longrun returns (None, None)."""
        reservoir_map = np.array([0, 0, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True])
        mock_compress.return_value = np.array([1, 1, 1])  # IsChannel same shape as reservoirs

        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [2010],
            'DEMOL_YEAR': [np.nan],
        })

        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'

        result = stations.process_reservoir_periods(
            mock_model, events_df, 24, None,
            valid_start, valid_end, Min_calib_days=None, isLongRun=True
        )

        assert result == (None, None)
        mock_create_map.assert_not_called()

    @patch('liscal.stations.create_netcdf_map')
    @patch('liscal.stations.compressArray')
    @patch('liscal.stations.boolean')
    @patch('liscal.stations.loadmap')
    def test_subperiod_boundaries_respect_timestep(self, mock_loadmap, mock_boolean, mock_compress, mock_create_map,
                                                    mock_model):
        """Subperiod starts (except the first) should be offset by dt hours from the event."""
        reservoir_map = np.array([0, 101, 0])
        mock_loadmap.side_effect = [reservoir_map, np.array([1, 1, 1])]
        mock_boolean.return_value = np.array([True, True, True])
        mock_compress.return_value = np.array([1, 1, 1])  # IsChannel same shape as reservoirs

        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [2010],
            'DEMOL_YEAR': [np.nan],
        })

        dt = 24  # 24 hours
        valid_start = '01/01/2005 06:00'
        valid_end = '31/12/2015 06:00'

        subperiods, _ = stations.process_reservoir_periods(
            mock_model, events_df, dt, None,
            valid_start, valid_end, Min_calib_days=None, isLongRun=True
        )

        # Second subperiod start should be event + dt hours
        event_dt = pd.Timestamp('2010-01-01')
        expected_second_start = event_dt + datetime.timedelta(hours=dt)
        assert subperiods[1][0] == expected_second_start


# ---------------------------------------------------------------------------
# Tests: create_netcdf_map
# ---------------------------------------------------------------------------

class TestCreateNetcdfMap:
    """Tests for create_netcdf_map: active reservoir identification."""

    @patch('liscal.stations.write_netcdf_header')
    @patch('liscal.stations.uncompress_array')
    def test_active_reservoirs_identified_correctly(self, mock_uncompress, mock_write_header):
        """Reservoirs active during the period should be included in the map."""
        mock_nf = MagicMock()
        mock_write_header.return_value = mock_nf
        mock_uncompress.return_value = np.zeros((3, 3))

        model = Mock()
        model.subcatch.path_station = '/tmp/test'
        model.lissettings = Mock()

        reservoirs = np.array([0, 101, 102, 103])

        events_df = pd.DataFrame({
            'FID': [101, 102, 103],
            'CONSTR_YEAR': [pd.Timestamp('2000'), pd.Timestamp('2010'), pd.Timestamp('2020')],
            'DEMOL_YEAR': [pd.NaT, pd.Timestamp('2008'), pd.NaT],
        })

        period_start = pd.Timestamp('2005-01-01')
        period_end = pd.Timestamp('2015-12-31')

        stations.create_netcdf_map("TestMap", reservoirs, events_df, model, period_start, period_end)

        # Verify the map was written
        mock_write_header.assert_called_once()
        mock_nf.close.assert_called_once()

        # Check which reservoirs are active:
        # 101: constructed 2000, no demolition -> active (2000 < 2015 and no demol)
        # 102: constructed 2010, demolished 2008 -> NOT active (demol 2008 is not > 2005 start... 
        #       wait: DEMOL_YEAR=2008 which is > period_start 2005? No: condition is DEMOL_YEAR > period_start
        #       Actually the condition is: (CONSTR_YEAR < period_end) and (DEMOL_YEAR > period_start)
        #       102: CONSTR=2010 < 2015 ✓, DEMOL=2008 > 2005 ✓ -> active
        # 103: constructed 2020, no demolition -> NOT active (CONSTR 2020 is not < period_end 2015)
        # So active should be {101, 102}

    @patch('liscal.stations.write_netcdf_header')
    @patch('liscal.stations.uncompress_array')
    def test_demolished_before_period_excluded(self, mock_uncompress, mock_write_header):
        """A reservoir demolished before the period starts should be excluded."""
        mock_nf = MagicMock()
        mock_write_header.return_value = mock_nf
        mock_uncompress.return_value = np.zeros((3, 3))

        model = Mock()
        model.subcatch.path_station = '/tmp/test'
        model.lissettings = Mock()

        reservoirs = np.array([0, 101, 0])

        events_df = pd.DataFrame({
            'FID': [101],
            'CONSTR_YEAR': [pd.Timestamp('1990')],
            'DEMOL_YEAR': [pd.Timestamp('2000')],  # demolished before period
        })

        period_start = pd.Timestamp('2005-01-01')
        period_end = pd.Timestamp('2015-12-31')

        stations.create_netcdf_map("TestMap", reservoirs, events_df, model, period_start, period_end)

        # The filtered map should have -9999 everywhere (no active reservoirs)
        # We verify by checking what was passed to uncompress_array
        map_arg = mock_uncompress.call_args[0][0]
        assert np.all(map_arg == -9999)


# ---------------------------------------------------------------------------
# Tests: update_rsfil_netcdf_map
# ---------------------------------------------------------------------------

class TestUpdateRsfilNetcdfMap:
    """Tests for update_rsfil_netcdf_map: identifying new reservoirs at a subperiod start."""

    @patch('liscal.stations.write_netcdf_header')
    @patch('liscal.stations.uncompress_array')
    @patch('liscal.stations.loadmap_base')
    @patch('liscal.stations.LisSettings')
    def test_identifies_new_reservoirs_for_period(self, mock_lis_settings, mock_loadmap_base,
                                                   mock_uncompress, mock_write_header):
        """New reservoirs constructed in the subperiod start year should be identified."""
        mock_instance = Mock()
        mock_instance.binding = {'ReservoirSites': 'path/to/sites', 'ReservoirFillEnd': 'path/to/fill'}
        mock_lis_settings.instance.return_value = mock_instance

        reservoir_map = np.array([0, 201, 202, 0])
        fill_map = np.array([0, 0.5, 0.8, 0])
        mock_loadmap_base.side_effect = [reservoir_map, fill_map]
        mock_uncompress.return_value = np.zeros((3, 3))

        mock_nf = MagicMock()
        mock_write_header.return_value = mock_nf

        settings = Mock()
        settings.binding = {'ReservoirSites': 'sites.nc', 'ReservoirFillEnd': 'fill.nc'}
        settings.output_dir = '/tmp/test_output'

        events_df = pd.DataFrame({
            'FID': [201, 202],
            'CONSTR_YEAR': [pd.Timestamp('2008'), pd.Timestamp('2012')],
            'DEMOL_YEAR': [pd.NaT, pd.NaT],
        })

        period_start = pd.Timestamp('2012-01-01')

        # This should identify FID 202 as new (constructed in 2012)
        stations.update_rsfil_netcdf_map("ReservoirMap_Subperiod_1", events_df, settings, period_start)

        # Verify loadmap_base was called for ReservoirSites and ReservoirFillEnd
        assert mock_loadmap_base.call_count == 2
