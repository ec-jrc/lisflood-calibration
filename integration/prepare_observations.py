import os
import pandas as pd
import argparse

def main(main_folder):
    # Define paths
    data_folder = os.path.join(main_folder, 'all_obs')
    corrected_folder = os.path.join(main_folder, 'corrected')
    missing_folder = os.path.join(main_folder, 'missing')
    output_folder = os.path.join(main_folder, 'output')
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the main data file
    data_file = os.path.join(data_folder, 'data.csv')
    data_df = pd.read_csv(data_file)
    
    # Process corrected files
    for filename in os.listdir(corrected_folder):
        if filename.endswith('.csv'):
            station_id = filename.split('.')[0]
            corrected_file = os.path.join(corrected_folder, filename)
            
            # Read the corrected data
            corrected_df = pd.read_csv(corrected_file)
            
            # Check if station_id exists in the data_df
            if station_id not in data_df.columns:
                print(f"Error: Station ID {station_id} not found in data.csv")
                continue

            # Check if the number of rows and day column match
            if len(data_df) != len(corrected_df) or not (data_df['day'].equals(corrected_df['day'])):
                print(f"Error: Mismatch in row count or 'day' values for station ID {station_id}")
                continue
            
            # Apply corrections
            data_df[station_id] = corrected_df[station_id]
    
    # Process missing files
    for filename in os.listdir(missing_folder):
        if filename.endswith('.csv'):
            station_id = filename.split('.')[0]
            missing_file = os.path.join(missing_folder, filename)
            
            # Read the missing data
            missing_df = pd.read_csv(missing_file)
            
            # Check if station_id already exists in the data_df
            if station_id in data_df.columns:
                print(f"Error: Station ID {station_id} already exists in data.csv")
                continue

            # Check if the number of rows and day column match
            if len(data_df) != len(missing_df) or not (data_df['day'].equals(missing_df['day'])):
                print(f"Error: Mismatch in row count or 'day' values for station ID {station_id}")
                continue
            
            # Add new station data
            data_df[station_id] = missing_df[station_id]

    # Format the 'day' column to 'dd/mm/yyyy HH:MM' with '00:00' as the time
    data_df['day'] = pd.to_datetime(data_df['day']).dt.strftime('%d/%m/%Y 00:00')

    # Save the corrected data to a new CSV file
    output_file = os.path.join(output_folder, 'data_corrected.csv')
    data_df.to_csv(output_file, index=False)
    print(f'Corrected data saved to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correct data using updates from corrected and missing files.')
    parser.add_argument('main_folder', type=str, help='The main folder containing data and corrected/missing subfolders.')
    args = parser.parse_args()

    main(args.main_folder)
