import os
import shutil
import argparse

def create_directories(base_path, continent, basin):
    path = os.path.join(base_path, continent, basin)
    os.makedirs(path, exist_ok=True)
    return path

def copy_and_modify_file(src, dest, replacements):
    with open(src, 'r') as file:
        content = file.read()

    for old, new in replacements.items():
        content = content.replace(old, new)

    with open(dest, 'w') as file:
        file.write(content)

def split_file_if_needed(file_path, max_rows=8, max_files=8):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_lines = len(lines)

    if num_lines > max_rows:
        num_files = min((num_lines + max_rows - 1) // max_rows, max_files)
        
        lines_per_file = num_lines // num_files
        extra_lines = num_lines % num_files

        start_index = 0
        split_files = []
        for i in range(num_files):
            end_index = start_index + lines_per_file + (1 if i < extra_lines else 0)
            split_file_path = f"{file_path.rsplit('.', 1)[0]}{i+1}.txt"
            with open(split_file_path, 'w') as split_file:
                split_file.writelines(lines[start_index:end_index])
            print(f"Created split file: {split_file_path}")
            start_index = end_index
            split_files.append(split_file_path)
        return split_files
    else:
        return [file_path]
            
def main(origin_base, calib_base, continent, basin):
    # Define paths based on the input arguments
    stations_base = os.path.join(calib_base, 'stations_GloFAS')
    catchments_base = os.path.join(calib_base, 'catchments')
    catchment_lists_base = os.path.join(calib_base, 'CatchmentsLists')
    templates_base = os.path.join(calib_base, 'templates_GloFAS')
    scripts_base = os.path.join(calib_base, 'scripts')

    # Create necessary directories
    stations_path = create_directories(stations_base, continent, basin)
    catchments_path = create_directories(catchments_base, continent, basin)

    # Copy stations list file
    stations_src = os.path.join(origin_base, f"GloFASv5_stationsselection_{continent}_{basin}", f"{continent}_{basin}_stationslist.txt")
    stations_dest = os.path.join(catchment_lists_base, f"CatchmentsToProcess_{continent}_{basin}.txt")
    shutil.copy2(stations_src, stations_dest)

    # Check and split the CatchmentsToProcess file if needed
    split_files = split_file_if_needed(stations_dest)

    # Read and modify calib_settings from a template file
    calib_settings_template = os.path.join(calib_base, 'calib_settings_templateGloFASv5.txt')
    calib_settings_file = os.path.join(calib_base, f"calib_settings_{continent}_{basin}GloFASv5.txt")

    replacements = {
        '<stations_data>': f"{stations_path}/stations_data.csv",
        '<stations_links>': f"{stations_path}/stations_links.csv",
        '<observed_discharges>': f"{origin_base}/GloFASv5_stationsselection_{continent}_{basin}/observations_GloFASv5_{continent}_{basin}_corrected.csv",
        '<return_periods>': f"{stations_path}/return_levels.nc",
        '<reservoir_events>': f"{calib_base}/GloFASv5_tables/reservoirs_glofas5_years_20250228.csv",
        '<param_ranges>': f"{calib_base}/GloFASv5_templates/param_ranges_v1.csv",
        '<subcatchment_path>': f"{catchments_path}",
        '<LFSettings>': f"{templates_base}/OSLisfloodGloFASv5calibration_v1_{continent}_{basin}.xml"
    }

    copy_and_modify_file(calib_settings_template, calib_settings_file, replacements)

    # Copy and modify XML template
    xml_src = os.path.join(templates_base, "OSLisfloodGloFASv5calibration_v1.xml")
    xml_dest = os.path.join(templates_base, f"OSLisfloodGloFASv5calibration_v1_{continent}_{basin}.xml")
    xml_replacements = {
        '$(PathRoot)/maps/waterregions__only.nc': f'$(PathRoot)/maps/waterregions_{continent}_{basin}_only.nc'
    }
    copy_and_modify_file(xml_src, xml_dest, xml_replacements)

    # Create shell script
    script_1to3_src  = os.path.join(scripts_base, "arun_cal_1to3template.sh")
    con_abbr = continent[:3].capitalize()
    script_dest = os.path.join(scripts_base, f"arun_cal_1to3{con_abbr}{basin}.sh")
    script_replacements = {
        'CONTINENT="template"': f'CONTINENT="{continent}"',
        'BASIN="template"': f'BASIN="{basin}"'
    }
    copy_and_modify_file(script_1to3_src , script_dest, script_replacements)

    # Create shell script(s) from arun_cal_4template.sh
    script_4_src = os.path.join(scripts_base, "arun_cal_4template.sh")
    
    for idx, split_file in enumerate(split_files):
        suffix = f"{idx+1}" if len(split_files) > 1 else ""
        script_dest = os.path.join(scripts_base, f"arun_cal_4{con_abbr}{basin}{suffix}.sh")
        n_file_value = f"{idx+1}" if len(split_files) > 1 else ""

        script_replacements = {
            'CONTINENT="template"': f'CONTINENT="{continent}"',
            'BASIN="template"': f'BASIN="{basin}"',
            'N_FILE="template"': f'N_FILE="{n_file_value}"'
        }
        copy_and_modify_file(script_4_src, script_dest, script_replacements)

    print(f"Setup completed for continent: {continent}, basin: {basin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup directories and files for a given continent and basin.")
    parser.add_argument('origin_base', type=str, help='Base path for origin files')
    parser.add_argument('calib_base', type=str, help='Base path for calibration files')
    parser.add_argument('continent', type=str, help='Name of the continent')
    parser.add_argument('basin', type=str, help='Name of the basin')

    args = parser.parse_args()
    main(args.origin_base, args.calib_base, args.continent, args.basin)
