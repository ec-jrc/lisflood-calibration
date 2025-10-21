import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def generate_days_column():
    """Generate a DataFrame with a 'day' column from 01/01/1980 to 01/01/2024."""
    start_date = datetime(1980, 1, 1)
    end_date = datetime(2024, 1, 1)
    days_count = (end_date - start_date).days + 1
    day_list = [(start_date + timedelta(days=i)).strftime('%d/%m/%Y 00:00') for i in range(days_count)]
    return pd.DataFrame({'day': day_list})

def main(main_folder, prefix):
    # Define paths with the prefix
    stations_selection_folder = os.path.join(main_folder, f'GloFASv5_stationsselection_{prefix}')
    observations_folder = os.path.join(main_folder, 'GloFASv5_Europe_observations_dataframe')
    output_folder = os.path.join(main_folder, 'output_corrected')
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the generate_observations_file.csv with prefix
    generate_file = os.path.join(stations_selection_folder, f'{prefix}_generate_observations_file.csv')
    generate_df = pd.read_csv(generate_file)
    
    # Create the base DataFrame with 'day' column
    corrected_df = generate_days_column()
    
    # Load NRT files once
    nrt_dfs = {}
    for part_file in ['Europe_NRT_part1.csv', 'Europe_NRT_part2.csv']:
        nrt_file = os.path.join(observations_folder, 'NRT', part_file)
        if os.path.exists(nrt_file):
            nrt_dfs[part_file] = pd.read_csv(nrt_file)
    
    # Load HIST file once
    hist_file = os.path.join(observations_folder, 'HIST', 'Europe_HIST.csv')
    hist_df = pd.read_csv(hist_file) if os.path.exists(hist_file) else None

    # Load FranceHIST file once
    francehist_file = os.path.join(observations_folder, 'FranceHIST', 'FranceHIST.csv')
    francehist_df = pd.read_csv(francehist_file) if os.path.exists(francehist_file) else None

    # Load efas Qts_calib_9023_daily.csv file once
    efas_file = os.path.join(observations_folder, 'efas', 'Qts_calib_9023_daily.csv')
    efas_df = pd.read_csv(efas_file,sep='\t') if os.path.exists(efas_file) else None    # this file has tabs to separate columns
    
    # Process each row in the generate_observations_file.csv
    for index, row in generate_df.iterrows():
        station_id = str(row['ID'])  # Ensure station_id is a string
        folder_value = row['folder']
        
        if folder_value == 'NRT':
            # Look for the station ID in loaded NRT DataFrames
            found = False
            for part_file, nrt_df in nrt_dfs.items():
                if station_id in nrt_df.columns:
                    # Check if 'day' columns match
                    if corrected_df['day'].equals(nrt_df['day']):
                        corrected_df[station_id] = nrt_df[station_id]
                    else:
                        print(f"Warning: Day values do not match for Station ID {station_id} in {part_file}.")
                    found = True
                    break
            if not found:
                print(f"Warning: Station ID {station_id} not found in NRT files.")
        
        elif folder_value == 'HIST':
            if hist_df is not None:
                if station_id in hist_df.columns:
                    # Check if 'day' columns match
                    if corrected_df['day'].equals(hist_df['day']):
                        corrected_df[station_id] = hist_df[station_id]
                    else:
                        print(f"Warning: Day values do not match for Station ID {station_id} in HIST.csv.")
                else:
                    print(f"Warning: Station ID {station_id} not found in HIST.csv.")
            else:
                print("Warning: Europe_HIST.csv not found.")

        elif folder_value == 'FranceHIST':
            if francehist_df is not None:
                if station_id in francehist_df.columns:
                    # Check if 'day' columns match
                    if corrected_df['day'].equals(francehist_df['day']):
                        corrected_df[station_id] = francehist_df[station_id]
                    else:
                        print(f"Warning: Day values do not match for Station ID {station_id} in FranceHIST.csv.")
                else:
                    print(f"Warning: Station ID {station_id} not found in FranceHIST.csv.")
            else:
                print("Warning: FranceHIST.csv not found.")

        elif folder_value == 'efas':
            if efas_df is not None:
                if station_id in efas_df.columns:
                    # Check if 'day' columns match
                    if corrected_df['day'].equals(efas_df['day']):
                        corrected_df[station_id] = efas_df[station_id]
                    else:
                        print(f"Warning: Day values do not match for Station ID {station_id} in efas Qts_calib_9023_daily.csv.")
                else:
                    print(f"Warning: Station ID {station_id} not found in efas Qts_calib_9023_daily.csv.")
            else:
                print("Warning: efas Qts_calib_9023_daily.csv not found.")
        
        else:
            # Look for the data in the specified folder under stations_selection_folder
            specific_folder_file = os.path.join(stations_selection_folder, folder_value, f"{station_id}.csv")
            if os.path.exists(specific_folder_file):
                specific_df = pd.read_csv(specific_folder_file)
                
                # Check if 'day' or 'time' columns exist
                if 'day' in specific_df.columns:
                    date_column = 'day'
                elif 'time' in specific_df.columns:
                    date_column = 'time'
                else:
                    print(f"Warning: No 'day' or 'time' column found for Station ID {station_id} in {folder_value}.")
                    continue

                # Check if day/time columns match
                if corrected_df['day'].equals(specific_df[date_column]):
                    corrected_df[station_id] = specific_df[station_id]
                else:
                    print(f"Warning: Day values do not match for Station ID {station_id} in {folder_value}.")
            else:
                print(f"Warning: File for Station ID {station_id} not found in specified folder {folder_value}.")

    # Save the corrected data to a new CSV file
    output_file = os.path.join(output_folder, f'observations_GloFASv5_{prefix}_corrected.csv')
    corrected_df.to_csv(output_file, index=False)
    print(f'Corrected data saved to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate corrected data using observations file.')
    parser.add_argument('main_folder', type=str, help='The main folder containing data and observation subfolders.')
    parser.add_argument('prefix', type=str, help='Prefix for the stations selection files.')
    args = parser.parse_args()

    main(args.main_folder, args.prefix)
