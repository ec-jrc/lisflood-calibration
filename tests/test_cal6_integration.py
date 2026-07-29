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

from liscal import config, subcatchment, templates, calibration, objective, hydro_model

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


class TestCAL6ParamFiltering:
    """Integration test for filter_param_ranges on real LISFLOOD model init.

    Verifies that parameters are correctly dropped based on actual subcatchment
    properties (no lakes, no MCT channels, warm climate) rather than mocked values.
    """

    def test_param_ranges_filtering_on_real_model(self, tmp_path):
        """Initialize LISFLOOD model for 7838 and verify correct param drops.

        Catchment 7838 has:
        - No lake sites in maps → LISFLOOD sets simulateLakes=False → LakeMultiplier dropped
        - No MCT channels in maps → LISFLOOD sets MCTRouting=False → CalChanMan3 dropped
        - min_TAvgS=5.477 > TempSnow=1.0 → SnowMeltCoef dropped (warm catchment)
        - use_aridity_index_check=0 (default) → TransSub kept
        """
        settings_file, tmp_catchments = _create_settings(tmp_path)

        cfg = config.ConfigCalibration(settings_file)
        obsid = 7838
        subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=None, create_links=True)
        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        lock_mgr = calibration.LockManager(cfg.num_cpus)
        obj = objective.ObjectiveKGE(cfg, subcatch)
        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
        model.init_run()

        # Verify LISFLOOD detected no lakes/MCT in subcatchment maps
        assert model.lissettings.options['simulateLakes'] == False, \
            "Expected simulateLakes=False (no lake sites in 7838 maps)"
        assert model.lissettings.options['MCTRouting'] == False, \
            "Expected MCTRouting=False (no MCT channels in 7838 maps)"
        assert model.lissettings.options['simulateReservoirs'] == True, \
            "Expected simulateReservoirs=True"

        # Run the filter
        cfg.filter_param_ranges_after_init(
            model_initialized=model,
            split_lake_params=cfg.deap_param.split_lake_params,
        )

        # Verify dropped parameters
        assert 'LakeMultiplier' not in cfg.param_ranges.index, \
            "LakeMultiplier should be dropped (no lakes in subcatchment)"
        assert 'CalChanMan3' not in cfg.param_ranges.index, \
            "CalChanMan3 should be dropped (no MCT routing in subcatchment)"
        assert 'SnowMeltCoef' not in cfg.param_ranges.index, \
            "SnowMeltCoef should be dropped (min_TAvgS=5.477 > TempSnow=1.0)"

        # Verify kept parameters
        assert 'TransSub' in cfg.param_ranges.index, \
            "TransSub should be kept (use_aridity_index_check disabled)"
        assert 'UpperZoneTimeConstant' in cfg.param_ranges.index
        assert 'LowerZoneTimeConstant' in cfg.param_ranges.index
        assert 'CalChanMan1' in cfg.param_ranges.index
        assert 'GwLoss' in cfg.param_ranges.index

        # Verify original_param_ranges preserved
        assert len(cfg.original_param_ranges) == 12, \
            f"Expected 12 original params, got {len(cfg.original_param_ranges)}"
        assert 'LakeMultiplier' in cfg.original_param_ranges.index
        assert 'SnowMeltCoef' in cfg.original_param_ranges.index
        assert 'CalChanMan3' in cfg.original_param_ranges.index

        # Final count: 12 - 3 dropped = 9
        assert len(cfg.param_ranges) == 9, \
            f"Expected 9 filtered params, got {len(cfg.param_ranges)}: {list(cfg.param_ranges.index)}"

    def test_param_ranges_filtering_with_aridity_check(self, tmp_path):
        """Verify TransSub is kept when aridity check enabled but min_AridIdx < 0.5.

        Catchment 7838 has min_AridIdx=0.365 (arid), so TransSub stays even
        with use_aridity_index_check=1.
        """
        settings_file, tmp_catchments = _create_settings(tmp_path)

        # Enable aridity check in settings
        with open(settings_file, 'r') as f:
            content = f.read()
        content = content.replace('[Main]', '[Main]\nuse_aridity_index_check = 1')
        with open(settings_file, 'w') as f:
            f.write(content)

        cfg = config.ConfigCalibration(settings_file)
        obsid = 7838
        subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=None, create_links=True)
        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        lock_mgr = calibration.LockManager(cfg.num_cpus)
        obj = objective.ObjectiveKGE(cfg, subcatch)
        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
        model.init_run()

        cfg.filter_param_ranges_after_init(
            model_initialized=model,
            split_lake_params=cfg.deap_param.split_lake_params,
        )

        # min_AridIdx=0.365 < 0.5 → TransSub kept even with check enabled
        assert 'TransSub' in cfg.param_ranges.index, \
            "TransSub should be kept (min_AridIdx=0.365 < 0.5, arid catchment)"
        assert cfg.use_aridity_index_check == True

    def test_param_ranges_filtering_transsub_dropped_humid(self, tmp_path):
        """Verify TransSub IS dropped when aridity check enabled and min_AridIdx >= 0.5.

        Modifies station_data.csv to simulate a humid catchment (min_AridIdx=0.6).
        """
        settings_file, tmp_catchments = _create_settings(tmp_path)

        # Enable aridity check
        with open(settings_file, 'r') as f:
            content = f.read()
        content = content.replace('[Main]', '[Main]\nuse_aridity_index_check = 1')
        with open(settings_file, 'w') as f:
            f.write(content)

        # Modify station_data.csv to have min_AridIdx >= 0.5
        station_data_file = os.path.join(tmp_catchments, '7838', 'station', 'station_data.csv')
        station_df = pd.read_csv(station_data_file, index_col=0)
        station_df.loc['min_AridIdx'] = 0.6
        station_df.to_csv(station_data_file)

        cfg = config.ConfigCalibration(settings_file)
        obsid = 7838
        subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=None, create_links=True)
        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        lock_mgr = calibration.LockManager(cfg.num_cpus)
        obj = objective.ObjectiveKGE(cfg, subcatch)
        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
        model.init_run()

        cfg.filter_param_ranges_after_init(
            model_initialized=model,
            split_lake_params=cfg.deap_param.split_lake_params,
        )

        # min_AridIdx=0.6 >= 0.5 → TransSub dropped
        assert 'TransSub' not in cfg.param_ranges.index, \
            "TransSub should be dropped (min_AridIdx=0.6 >= 0.5, humid catchment)"

        # Still 8 params left (9 - TransSub)
        assert len(cfg.param_ranges) == 8, \
            f"Expected 8 filtered params, got {len(cfg.param_ranges)}: {list(cfg.param_ranges.index)}"

def _create_settings_with_full_params(tmp_path):
    """Create settings using param_ranges_full.csv (includes reservoir params)."""
    tmp_catchments = str(tmp_path / 'catchments')
    shutil.copytree(os.path.join(INPUT_DIR, 'catchments'), tmp_catchments)

    # Resolve XML template
    xml_src = os.path.join(INPUT_DIR, 'templates', 'OSLisfloodGloFASv5calibration_v1.xml')
    xml_dst = str(tmp_path / 'template.xml')
    xml_content = open(xml_src).read()
    xml_content = xml_content.replace('{TABLES_DIR}/', TABLES_DIR + '/')
    with open(xml_dst, 'w') as f:
        f.write(xml_content)

    # Resolve settings file with full param_ranges
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
    # Point to full param_ranges (includes ReservoirFloodStorage/OutflowFactor)
    content = content.replace(
        f'param_ranges = {INPUT_DIR}/templates/param_ranges_v1.csv',
        f'param_ranges = {INPUT_DIR}/templates/param_ranges_full.csv',
    )
    settings_file = str(tmp_path / 'settings.txt')
    with open(settings_file, 'w') as f:
        f.write(content)

    return settings_file, tmp_catchments


class TestCAL6ReservoirParamFiltering:
    """Integration test for reservoir parameter filtering on real LISFLOOD model.

    Uses param_ranges_full.csv which includes ReservoirFloodStorage and
    ReservoirFloodOutflowFactor to verify they are correctly kept or dropped
    based on the actual subcatchment data.
    """

    def test_reservoir_params_kept_when_reservoirs_present(self, tmp_path):
        """Reservoir params KEPT when subcatchment has reservoir sites.

        Catchment 7838 has reservoir data → LISFLOOD sets simulateReservoirs=True
        → ReservoirFloodStorage and ReservoirFloodOutflowFactor stay.
        """
        settings_file, tmp_catchments = _create_settings_with_full_params(tmp_path)

        cfg = config.ConfigCalibration(settings_file)
        obsid = 7838
        subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=None, create_links=True)
        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        lock_mgr = calibration.LockManager(cfg.num_cpus)
        obj = objective.ObjectiveKGE(cfg, subcatch)
        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
        model.init_run()

        # Confirm LISFLOOD found reservoirs in maps
        assert model.lissettings.options['simulateReservoirs'] == True

        cfg.filter_param_ranges_after_init(
            model_initialized=model,
            split_lake_params=cfg.deap_param.split_lake_params,
        )

        # Reservoir params should be kept
        assert 'ReservoirFloodStorage' in cfg.param_ranges.index, \
            "ReservoirFloodStorage should be kept (reservoir sites in subcatchment)"
        assert 'ReservoirFloodOutflowFactor' in cfg.param_ranges.index, \
            "ReservoirFloodOutflowFactor should be kept (reservoir sites in subcatchment)"

        # Full set minus Lake/MCT/Snow = 14 - 3 = 11
        assert len(cfg.param_ranges) == 11, \
            f"Expected 11 filtered params, got {len(cfg.param_ranges)}: {list(cfg.param_ranges.index)}"

    def test_reservoir_params_dropped_when_reservoirs_off_in_xml(self, tmp_path):
        """Reservoir params DROPPED when simulateReservoirs is OFF in XML.

        Forces simulateReservoirs=0 in the template to simulate a subcatchment
        configured without reservoirs from the start.
        """
        settings_file, tmp_catchments = _create_settings_with_full_params(tmp_path)

        # Patch the XML template: set simulateReservoirs to 0
        xml_dst = str(tmp_path / 'template.xml')
        with open(xml_dst, 'r') as f:
            xml_content = f.read()
        xml_content = xml_content.replace(
            '<setoption choice="1" name="simulateReservoirs"/>',
            '<setoption choice="0" name="simulateReservoirs"/>',
        )
        with open(xml_dst, 'w') as f:
            f.write(xml_content)

        cfg = config.ConfigCalibration(settings_file)
        obsid = 7838
        subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=None, create_links=True)
        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        lock_mgr = calibration.LockManager(cfg.num_cpus)
        obj = objective.ObjectiveKGE(cfg, subcatch)
        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
        model.init_run()

        # Confirm LISFLOOD has reservoirs OFF
        assert model.lissettings.options['simulateReservoirs'] == False

        cfg.filter_param_ranges_after_init(
            model_initialized=model,
            split_lake_params=cfg.deap_param.split_lake_params,
        )

        # Reservoir params should be dropped
        assert 'ReservoirFloodStorage' not in cfg.param_ranges.index, \
            "ReservoirFloodStorage should be dropped (simulateReservoirs=False)"
        assert 'ReservoirFloodOutflowFactor' not in cfg.param_ranges.index, \
            "ReservoirFloodOutflowFactor should be dropped (simulateReservoirs=False)"

        # Full set minus Lake/MCT/Snow/2xReservoir = 14 - 5 = 9
        assert len(cfg.param_ranges) == 9, \
            f"Expected 9 filtered params, got {len(cfg.param_ranges)}: {list(cfg.param_ranges.index)}"

    def test_reservoir_params_dropped_when_no_reservoirs_in_maps(self, tmp_path):
        """Reservoir params DROPPED when simulateReservoirs=ON but no reservoir sites in maps.

        The XML has simulateReservoirs=1, but after LISFLOOD init finds the
        reservoir map has no valid sites, it switches simulateReservoirs to False.
        This simulates a subcatchment that was cut from a region without reservoirs.
        """
        import xarray as xr

        settings_file, tmp_catchments = _create_settings_with_full_params(tmp_path)

        # Replace reservoir site IDs with 0 (no reservoirs) while keeping the spatial mask
        res_map_path = os.path.join(
            tmp_catchments, '7838', 'maps', '20250307_reservoirs_Global_03min.nc'
        )
        ds = xr.open_dataset(res_map_path)
        # Set all valid (non-NaN) reservoir IDs to 0 — means no reservoir at that pixel
        band = ds['Band1'].values
        band[~np.isnan(band)] = 0.0
        ds['Band1'].values = band
        ds.to_netcdf(res_map_path + '.tmp')
        ds.close()
        os.replace(res_map_path + '.tmp', res_map_path)

        cfg = config.ConfigCalibration(settings_file)
        obsid = 7838
        subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=None, create_links=True)
        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        lock_mgr = calibration.LockManager(cfg.num_cpus)
        obj = objective.ObjectiveKGE(cfg, subcatch)
        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
        model.init_run()

        # LISFLOOD should detect no reservoir sites and switch option OFF
        assert model.lissettings.options['simulateReservoirs'] == False, \
            "Expected simulateReservoirs=False (reservoir map zeroed out, no sites)"

        cfg.filter_param_ranges_after_init(
            model_initialized=model,
            split_lake_params=cfg.deap_param.split_lake_params,
        )

        # Reservoir params should be dropped
        assert 'ReservoirFloodStorage' not in cfg.param_ranges.index, \
            "ReservoirFloodStorage should be dropped (no reservoirs in maps)"
        assert 'ReservoirFloodOutflowFactor' not in cfg.param_ranges.index, \
            "ReservoirFloodOutflowFactor should be dropped (no reservoirs in maps)"

        # Full set minus Lake/MCT/Snow/2xReservoir = 14 - 5 = 9
        assert len(cfg.param_ranges) == 9, \
            f"Expected 9 filtered params, got {len(cfg.param_ranges)}: {list(cfg.param_ranges.index)}"
