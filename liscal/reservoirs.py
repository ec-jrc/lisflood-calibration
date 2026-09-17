"""
Reservoir-related logic for the calibration tool.

This module consolidates all reservoir management code:
- Filtering observation periods based on reservoir construction/demolition events
- Generating reservoir NetCDF maps for calibration subperiods
- Warm-start settings file generation for long-term runs with subperiods
- Merging TSS files from reservoir warm-start runs
- Updating reservoir fill maps between subperiods

Design rationale: lisflood and PCRaster are heavy dependencies that don't work
on all platforms (e.g., macOS). By keeping all lisflood/PCRaster imports here
and in hydro_model.py, the rest of the calibration tool remains testable
without those dependencies.
"""

import os
import datetime
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# Lisflood / PCRaster imports kept isolated here
from lisflood.global_modules.add1 import loadmap, loadmap_base, compressArray
from lisflood.global_modules.netcdf import uncompress_array, write_netcdf_header
from lisflood.global_modules.settings import LisSettings
from pcraster import boolean

from liscal import utils


def process_reservoir_periods(model_initialized, reservoir_events_df, dt,
                              observations_filtered, valid_start, valid_end,
                              Min_calib_days, isLongRun=False):
    """
    Process reservoir events to determine valid observation periods or subperiods.

    For calibration (isLongRun=False): finds the best observation period that
    avoids reservoir construction/demolition events, and generates a filtered
    reservoir map for that period.

    For long-term runs (isLongRun=True): splits the run into subperiods
    separated by reservoir events, and generates a reservoir map for each.

    Parameters
    ----------
    model_initialized : HydrologicalModel
        Initialized model with LISFLOOD settings loaded.
    reservoir_events_df : pd.DataFrame
        DataFrame with columns 'FID', 'CONSTR_YEAR', 'DEMOL_YEAR'.
    dt : int
        Time step in hours (6 or 24).
    observations_filtered : pd.Series or None
        Filtered observations (only needed for calibration, not long-term run).
    valid_start : str
        Start date string in '%d/%m/%Y %H:%M' format.
    valid_end : str
        End date string in '%d/%m/%Y %H:%M' format.
    Min_calib_days : float or None
        Minimum calibration days (only needed for calibration).
    isLongRun : bool
        If True, compute subperiods for warm-start long-term run.

    Returns
    -------
    If isLongRun=False:
        tuple(str, str) : best_period_start, best_period_end
    If isLongRun=True:
        tuple(list or None, pd.DataFrame or None) : subperiods, filtered_reservoir_events_df
    """
    # copy dates as string
    best_period_start, best_period_end = valid_start, valid_end

    # Ensure valid_start and valid_end are datetime objects
    valid_start = pd.to_datetime(valid_start, format="%d/%m/%Y %H:%M")
    valid_end = pd.to_datetime(valid_end, format="%d/%m/%Y %H:%M")

    # Check if reservoirs are simulated and load necessary maps
    if model_initialized.lissettings.options['simulateReservoirs']:
        reservoirs = loadmap('ReservoirSites')
        IsChannelPcr = boolean(loadmap('Channels', pcr=True))
        IsChannel = np.bool_(compressArray(IsChannelPcr))
        reservoirs[(reservoirs < 1) | (IsChannel == 0)] = 0

        # Get active reservoir sites
        ReservoirSitesCC = np.compress(reservoirs > 0, reservoirs)

        if ReservoirSitesCC.size > 0:
            # filter reservoir events data
            reservoir_events_df = reservoir_events_df[reservoir_events_df['FID'].isin(ReservoirSitesCC)]

            # Convert year columns to datetime
            reservoir_events_df['CONSTR_YEAR'] = pd.to_datetime(reservoir_events_df['CONSTR_YEAR'], format='%Y', errors='coerce')
            reservoir_events_df['DEMOL_YEAR'] = pd.to_datetime(reservoir_events_df['DEMOL_YEAR'], format='%Y', errors='coerce')

            # Gather potentially impacting events
            reservoir_events = set(
                [date for date in reservoir_events_df['CONSTR_YEAR'].dropna().tolist() +
                 reservoir_events_df['DEMOL_YEAR'].dropna().tolist()
                 if valid_start < date < valid_end]
            )
            reservoir_events = sorted(reservoir_events)

            if isLongRun:
                # Define subperiods from valid_start to valid_end interrupted by events
                start_date = valid_start
                subperiods = []
                for event in reservoir_events + [valid_end]:
                    if start_date < event:
                        if start_date != valid_start:
                            start_date += datetime.timedelta(hours=int(dt))
                    subperiods.append((start_date, event))
                    start_date = event

                # Generate NetCDF map for each subperiod
                for idx, (sub_start, sub_end) in enumerate(subperiods):
                    map_name = f"ReservoirMap_Subperiod_{idx}"
                    create_netcdf_map(map_name, reservoirs, reservoir_events_df, model_initialized, sub_start, sub_end)
                return subperiods, reservoir_events_df
            else:
                # Determine the most recent valid observation period
                min_steps = int(Min_calib_days * 24 / dt)
                last_valid_end = valid_end
                best_period_start_dt = None
                best_period_end_dt = None
                for event in reversed([valid_start - datetime.timedelta(hours=int(dt))] + reservoir_events):
                    period_observations = observations_filtered.copy()
                    period_observations.index = pd.to_datetime(period_observations.index, format='%d/%m/%Y %H:%M')
                    period_observations = period_observations[event:last_valid_end]
                    if len(period_observations) >= min_steps:
                        best_period_start_dt, best_period_end_dt = period_observations.index[0], period_observations.index[-1]
                        best_period_start, best_period_end = best_period_start_dt.strftime('%d/%m/%Y %H:%M'), best_period_end_dt.strftime('%d/%m/%Y %H:%M')
                        break
                    last_valid_end = event - datetime.timedelta(hours=int(dt))

                if best_period_start_dt is None or best_period_end_dt is None:
                    raise Exception('Error: unable to find best period with {} steps after reservoir events check.'.format(min_steps))
                map_name = "FilteredReservoirMap"
                create_netcdf_map(map_name, reservoirs, reservoir_events_df, model_initialized, best_period_start_dt, best_period_end_dt)

    if isLongRun:
        return None, None  # If isLongRun, no specific period to return
    return best_period_start, best_period_end


def create_netcdf_map(map_name, reservoirs, reservoir_events_df, model_initialized, period_start_dt, period_end_dt):
    """
    Create a NetCDF map of active reservoirs for a given time period.

    Parameters
    ----------
    map_name : str
        Name for the output map variable and file.
    reservoirs : np.ndarray
        Compressed reservoir sites array from LISFLOOD.
    reservoir_events_df : pd.DataFrame
        DataFrame with reservoir events (FID, CONSTR_YEAR, DEMOL_YEAR).
    model_initialized : HydrologicalModel
        Initialized model with settings.
    period_start_dt : datetime
        Start of the period.
    period_end_dt : datetime
        End of the period.
    """
    # Identify active reservoirs for the selected period
    active_reservoirs = {
        row['FID'] for _, row in reservoir_events_df.iterrows()
        if (pd.isna(row['CONSTR_YEAR']) or row['CONSTR_YEAR'] < period_end_dt) and
           (pd.isna(row['DEMOL_YEAR']) or row['DEMOL_YEAR'] > period_start_dt)
    }

    # Save the reservoir filtered map for the selected period
    FilteredReservoirMap = np.full_like(reservoirs, -9999, dtype=float)
    for res_id in active_reservoirs:
        FilteredReservoirMap[reservoirs == res_id] = res_id

    strFilteredReservoirMap = os.path.join(model_initialized.subcatch.path_station, f'{map_name}.nc')
    nf1 = write_netcdf_header(model_initialized.lissettings, map_name, strFilteredReservoirMap, None,
                              map_name, map_name, "",
                              None, None, None)

    map_np = uncompress_array(FilteredReservoirMap)
    nf1.variables[map_name][:, :] = map_np
    nf1.close()
    print("Generated new ReservoirSites content to:", strFilteredReservoirMap)


def update_rsfil_netcdf_map(map_name, reservoir_events_df, settings, period_start_dt):
    """
    Update the ReservoirFill NetCDF map for new reservoirs in a subperiod.

    Sets the fill state to 0.1 for reservoirs whose construction year matches
    the start of the current subperiod.

    Parameters
    ----------
    map_name : str
        Name of the reservoir fill map (without .nc extension).
    reservoir_events_df : pd.DataFrame
        DataFrame with reservoir events.
    settings : LisSettings
        LISFLOOD settings for the current run.
    period_start_dt : datetime
        Start date of the current subperiod.
    """
    settings_instance = LisSettings.instance()
    settings_instance.binding['ReservoirSites'] = settings.binding['ReservoirSites']
    settings_instance.binding['ReservoirFillEnd'] = settings.binding['ReservoirFillEnd']
    reservoirs = loadmap_base('ReservoirSites')
    try:
        ReservoirFillMap = loadmap_base('ReservoirFillEnd', force_load_with_nans=True)
    except:
        # in case map doesn't exist, initialize an empty map
        ReservoirFillMap = np.full_like(reservoirs, 0, dtype=float)

    # Identify new reservoirs for the selected period
    new_reservoirs = {
        row['FID'] for _, row in reservoir_events_df.iterrows()
        if (row['CONSTR_YEAR'].year == period_start_dt.year)
    }

    # Update the reservoir fill map for new reservoirs
    for res_id in new_reservoirs:
        ReservoirFillMap[reservoirs == res_id] = 0.1  # set new reservoir fill state to 0.1

    strReservoirFillMap = os.path.join(settings.output_dir, f'{map_name}.nc')
    nf1 = write_netcdf_header(settings, map_name, strReservoirFillMap, None,
                              map_name, map_name, "",
                              None, None, None)

    map_np = uncompress_array(ReservoirFillMap)
    nf1.variables[map_name][:, :] = map_np
    nf1.close()
    print("Generated new ReservoirFill content to:", strReservoirFillMap)


def inject_filtered_reservoir_map(out_xml, path_station, reservoir_events_config):
    """
    Inject the FilteredReservoirMap path into the LISFLOOD XML settings
    if a filtered map exists for the station.

    This is used during calibration runs (not long-term) to replace the
    default ReservoirSites map with a filtered one that only contains
    reservoirs active during the calibration period.

    Parameters
    ----------
    out_xml : str
        XML content as string.
    path_station : str
        Path to the station directory.
    reservoir_events_config : str or None
        Path to reservoir events CSV (from cfg.reservoir_events).

    Returns
    -------
    str
        Modified XML content.
    """
    if reservoir_events_config is not None:
        strFilteredReservoirMap = os.path.join(path_station, 'FilteredReservoirMap.nc')
        if os.path.exists(strFilteredReservoirMap):
            root = ET.fromstring(out_xml)
            reservoir_sites_element = root.find(".//textvar[@name='ReservoirSites']")
            if reservoir_sites_element is not None:
                reservoir_sites_element.set("value", strFilteredReservoirMap)
                print("Updated ReservoirSites content to:", strFilteredReservoirMap)
                out_xml = ET.tostring(root, encoding="unicode")
    return out_xml


def write_warmstart_settings_files(lis_template, run_id, original_run_file, path_station,
                                   subperiods, includeLakes, includeMCT):
    """
    Generate warm-start settings files for each subperiod of a long-term run.

    Each subperiod gets its own XML settings file with updated dates,
    reservoir maps, and warm-start initial conditions from the previous period.

    Parameters
    ----------
    lis_template : LisfloodSettingsTemplate
        Template object (used for settings_path method).
    run_id : str
        Run identifier.
    original_run_file : str
        Path to the original run settings XML file.
    path_station : str
        Path to the station directory containing reservoir maps.
    subperiods : list of tuple
        List of (start_datetime, end_datetime) for each subperiod.
    includeLakes : bool
        Whether lake state variables should be included in warm-start.
    includeMCT : bool
        Whether MCT routing state variables should be included in warm-start.

    Returns
    -------
    list of str
        Paths to the generated warm-start settings files.
    """
    textvar_end_mappings = {
        "OFDirectInitValue": "OFDirectEnd",
        "OFOtherInitValue": "OFOtherEnd",
        "OFForestInitValue": "OFForestEnd",
        "SnowCoverAInitValue": "SnowCoverAEnd",
        "SnowCoverBInitValue": "SnowCoverBEnd",
        "SnowCoverCInitValue": "SnowCoverCEnd",
        "FrostIndexInitValue": "FrostIndexEnd",
        "CumIntInitValue": "CumInterceptionEnd",
        "UZInitValue": "UZEnd",
        "DSLRInitValue": "DSLREnd",
        "LZInitValue": "LZEnd",
        "TotalCrossSectionAreaInitValue": "ChanCrossSectionEnd",
        "ThetaInit1Value": "Theta1End",
        "ThetaInit2Value": "Theta2End",
        "ThetaInit3Value": "Theta3End",
        "CrossSection2AreaInitValue": "CrossSection2End",
        "PrevSideflowInitValue": "ChSideEnd",
        "CumIntForestInitValue": "CumInterceptionForestEnd",
        "UZForestInitValue": "UZForestEnd",
        "DSLRForestInitValue": "DSLRForestEnd",
        "ThetaForestInit1Value": "Theta1ForestEnd",
        "ThetaForestInit2Value": "Theta2ForestEnd",
        "ThetaForestInit3Value": "Theta3ForestEnd",
        "CumIntIrrigationInitValue": "CumInterceptionIrrigationEnd",
        "UZIrrigationInitValue": "UZIrrigationEnd",
        "DSLRIrrigationInitValue": "DSLRIrrigationEnd",
        "ThetaIrrigationInit1Value": "Theta1IrrigationEnd",
        "ThetaIrrigationInit2Value": "Theta2IrrigationEnd",
        "ThetaIrrigationInit3Value": "Theta3IrrigationEnd",
        "CumIntSealedInitValue": "CumIntSealedEnd",
        "ReservoirInitialFill": "ReservoirFillEnd",
        "PrevDischarge": "ChanQEnd",
        "PrevDischargeAvg": "ChanQAvgDtEnd"
    }
    if includeLakes is True:
        textvar_end_mappings.update({
            "LakeInitialLevelValue": "LakeLevelEnd",
            "LakePrevInflowValue": "LakePrevInflowEnd",
            "LakePrevOutflowValue": "LakePrevOutflowEnd"
        })
    if includeMCT is True:
        textvar_end_mappings.update({
            "PrevCmMCTInitValue": "PrevCmMCTEnd",
            "PrevDmMCTInitValue": "PrevDmMCTEnd"
        })

    with open(original_run_file, "r") as f:
        out_xml = f.read()
    warmstart_run_files = []

    # Generate settings file for each subperiod
    last_sub_end = None  # start without setting timestepInit
    for idx, (sub_start, sub_end) in enumerate(subperiods):
        warmstart_run_file = lis_template.settings_path('Run', run_id, idx)
        strFilteredReservoirMap = os.path.join(path_station, f"ReservoirMap_Subperiod_{idx}.nc")
        if os.path.exists(strFilteredReservoirMap):
            # Parse the XML from the string
            root = ET.fromstring(out_xml)
            lfuser_section = root.find(".//lfuser")

            # Get DtSec value to get the timestep
            dtsec_element = lfuser_section.find(".//textvar[@name='DtSec']")
            timestepInit_element = lfuser_section.find(".//textvar[@name='timestepInit']")

            ColdStart_element = root.find(".//setoption[@name='ColdStart']")
            repEndMaps_element = root.find(".//setoption[@name='repEndMaps']")

            # Find the element with the tag 'ReservoirSites'
            reservoir_sites_element = lfuser_section.find(".//textvar[@name='ReservoirSites']")
            if reservoir_sites_element is None:
                reservoir_sites_element = root.find(".//textvar[@name='ReservoirSites']")

            # Find elements with the tag 'StepStart' and 'StepEnd'
            step_start_element = lfuser_section.find(".//textvar[@name='StepStart']")
            step_end_element = lfuser_section.find(".//textvar[@name='StepEnd']")

            # Find element MapsCaching: we want to disable it for the longrun warmstart
            maps_caching_element = lfuser_section.find(".//textvar[@name='MapsCaching']")

            # Check if elements are found and update content
            if (reservoir_sites_element is not None) and \
                    (dtsec_element is not None) and \
                    (step_start_element is not None) and \
                    (timestepInit_element is not None) and \
                    (ColdStart_element is not None) and \
                    (repEndMaps_element is not None) and \
                    (step_end_element is not None) and \
                    (maps_caching_element is not None):

                # disable MapsCaching for the longrun warmstart
                maps_caching_element.set("value", "False")

                # we need end maps to run the Warm Start
                repEndMaps_element.set("choice", "1")

                dtsec_value = int(dtsec_element.get('value'))  # Convert to integer
                reservoir_sites_element.set("value", strFilteredReservoirMap)

                str_sub_start = sub_start.strftime('%d/%m/%Y %H:%M')
                str_sub_end = sub_end.strftime('%d/%m/%Y %H:%M')
                step_start_element.set("value", str_sub_start)
                step_end_element.set("value", str_sub_end)
                print(f"Updated Period and ReservoirSites content to: {str_sub_start}, {str_sub_end}, {strFilteredReservoirMap} in {warmstart_run_file}")

                if idx > 0:
                    assert (last_sub_end is not None)
                    timestepInit_element.set("value", last_sub_end.strftime('%d/%m/%Y %H:%M'))
                    ColdStart_element.set("choice", "0")

                    n = 0
                    for init_name, end_name in textvar_end_mappings.items():
                        # Find the InitValue element
                        init_element = lfuser_section.find(f".//textvar[@name='{init_name}']")
                        if init_element is None:
                            init_element = root.find(f".//textvar[@name='{init_name}']")
                        # Find the corresponding End element
                        end_element = lfuser_section.find(f".//textvar[@name='{end_name}']")
                        if end_element is None:
                            end_element = root.find(f".//textvar[@name='{end_name}']")

                        if init_element is not None and end_element is not None:
                            n += 1
                            init_element.set('value', end_element.get('value'))
                        else:
                            raise Exception(f'Missing Init element {init_name} or End element {end_name} in XML file for the longrun warmstart')

                    print(f"Updated {n} Init Vars for warm start in {warmstart_run_file}")

                out_xml = ET.tostring(root, encoding="unicode")

                # update last sub_end for the next timestepInit value
                last_sub_end = sub_end
            else:
                raise Exception('Missing element in XML file for the longrun warmstart')
        with open(warmstart_run_file, "w") as f:
            f.write(out_xml)
        warmstart_run_files.append(warmstart_run_file)

    return warmstart_run_files


def merge_tss_files(tss_file_list, output_tss_file):
    """
    Merge multiple TSS files from reservoir warm-start subperiod runs.

    Concatenates data from TSS files produced by successive warm-start runs
    into a single output file, handling different column sets across files.

    Parameters
    ----------
    tss_file_list : list of str
        Paths to the TSS files to merge.
    output_tss_file : str
        Path for the merged output TSS file.
    """
    concatenated_data = []
    all_column_names = set()
    file_data = []

    for file_path in tss_file_list:
        if not os.path.exists(file_path):
            print(f'{file_path} not found. Skipping...')
        else:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Remove empty lines at the end of the file
            lines = [line for line in lines if line.strip()]

            # Parse header to determine metadata lines and column count
            header_lines, data_start_index, n_columns, column_names = _parse_tss_header(lines)
            all_column_names.update(column_names)
            file_data.append((header_lines, data_start_index, column_names, lines[data_start_index:]))

    if len(file_data) > 0:
        # Sort column names to have a consistent order
        all_column_names = sorted(all_column_names)

        # Build the final merged data
        merged_header = _create_merged_header(file_data[0][0], all_column_names)
        concatenated_data.extend(merged_header)

        for _, data_start_index, column_names, data_lines in file_data:
            column_index_map = {name: i for i, name in enumerate(column_names)}
            for line in data_lines:
                parts = line.split()
                timestep = parts[0]
                values = parts[1:]
                merged_line = [f"{timestep:>9}"]  # Format the timestep with width of 9

                for col_name in all_column_names:
                    if col_name in column_index_map:
                        value_index = column_index_map[col_name]
                        merged_line.append(f"{float(values[value_index]):>14}")  # Format values with width of 14
                    else:
                        merged_line.append(f"{np.nan:>14}")  # Fill missing columns with nan

                concatenated_data.append(' '.join(merged_line) + '\n')

        # Write the concatenated data to the output file
        with open(output_tss_file, 'w') as output_file:
            output_file.writelines(concatenated_data)

        print(f'Merged file created: {output_tss_file}')


def _parse_tss_header(lines):
    """Parse TSS file header to extract column information."""
    n_columns = int(lines[1].strip()) - 1  # exclude timestep from columns
    column_names = [lines[i].strip() for i in range(3, 3 + n_columns)]
    data_start_index = 3 + n_columns
    return lines[:data_start_index], data_start_index, n_columns, column_names


def _create_merged_header(header_lines, all_column_names):
    """Create a merged TSS header with all column names."""
    merged_header = header_lines[:3]  # Keep the first three lines
    merged_header[1] = f"{len(all_column_names) + 1}\n"  # Update column count + timestep
    merged_header.extend([name + '\n' for name in all_column_names])
    return merged_header


def run_longterm_with_subperiods(subcatch, lis_template, run_id, run_file,
                                 subperiods, filtered_reservoir_events,
                                 includeLakes, includeMCT):
    """
    Execute a long-term LISFLOOD run with warm-start subperiods.

    Generates warm-start settings files, runs LISFLOOD for each subperiod,
    updates reservoir fill maps between periods, and merges output TSS files.

    Parameters
    ----------
    subcatch : SubCatchment
        Subcatchment object.
    lis_template : LisfloodSettingsTemplate
        Template for settings files.
    run_id : str
        Run identifier.
    run_file : str
        Path to the original run settings file.
    subperiods : list of tuple
        List of (start_datetime, end_datetime) for each subperiod.
    filtered_reservoir_events : pd.DataFrame
        Filtered reservoir events DataFrame.
    includeLakes : bool
        Whether lake options are enabled.
    includeMCT : bool
        Whether MCT routing is enabled.
    """
    import lisf1

    # generate subperiods settings files
    warmstart_run_files = write_warmstart_settings_files(
        lis_template, run_id, run_file, subcatch.path_station,
        subperiods, includeLakes, includeMCT
    )

    # run different periods with warm start
    list_of_variable_to_merge = None
    var_files = {}
    original_var_tss_name = {}
    for idx, warmstart_run_file in enumerate(warmstart_run_files):
        settings = LisSettings(warmstart_run_file, "")
        if idx > 0:
            rsfil_map_name = settings.binding['ReservoirFillEnd']
            rsfil_map_name = rsfil_map_name[:-3] if rsfil_map_name.lower().endswith('.nc') else rsfil_map_name
            # copy the ReservoirFill end map to a backup before editing it for the next run
            cmd = f'cp {rsfil_map_name}.nc {rsfil_map_name}_ws_{idx - 1}.nc'
            utils.run_cmd(cmd)
            sub_start, _ = subperiods[idx]
            map_name = os.path.basename(rsfil_map_name)  # map_name will not contain ".nc"
            update_rsfil_netcdf_map(map_name, filtered_reservoir_events, settings, sub_start)
        else:
            list_of_variable_to_merge = [k for k in settings.report_timeseries.keys()]
            for varName in list_of_variable_to_merge:
                var_files[varName] = []
                original_var_tss_name[varName] = None
        lisf1.main(warmstart_run_file, '-q')
        # rename tss files to keep info for the final merge
        for varName in list_of_variable_to_merge:
            original_var_tss_name[varName] = settings.binding[varName]
            var_tss_name = original_var_tss_name[varName][:-4] if original_var_tss_name[varName].lower().endswith('.tss') else original_var_tss_name[varName]
            var_tss_name_dest = f'{var_tss_name}_ws_{idx}.tss'
            cmd = f'mv {var_tss_name}.tss {var_tss_name_dest}'
            utils.run_cmd(cmd)
            var_files[varName].append(var_tss_name_dest)

    # merge tss files from all subperiods
    for varName in list_of_variable_to_merge:
        merge_tss_files(var_files[varName], original_var_tss_name[varName])
