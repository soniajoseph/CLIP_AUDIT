import pandas as pd
import numpy as np

import json
from tqdm.auto import tqdm


# def process_and_save_activation_data(input_data, output_file):
#     """
#     Process activation data, calculate cumulative sums, and save to a JSON file.

#     :param input_data: A string of CSV data or a path to a CSV file
#     :param output_file: Path to save the processed data as JSON
#     """
#     # Read the data
#     try:
#         df = pd.read_csv(input_data)
#     except:
#         df = pd.read_csv(pd.compat.StringIO(input_data))

#     # Calculate cumulative sum and percentiles
#     df['Cumulative_Count'] = df['Count'].cumsum()
#     total_count = df['Count'].sum()
#     df['Percentile'] = df['Cumulative_Count'] / total_count * 100

#     # Prepare data for JSON serialization
#     data_to_save = {
#         'data': df.to_dict(orient='records'),
#         'total_count': int(total_count)
#     }

#     # Save to JSON file
#     with open(output_file, 'w') as f:
#         json.dump(data_to_save, f)

#     print(f"Processed data saved to {output_file}")



# for layer in tqdm(range(12)):
#     input_path = f'../histograms/blocks.{layer}.mlp.hook_post_percentiles.txt'
#     output_file = f'../histograms/blocks.{layer}.mlp.hook_post_cumulative_sum.json'
#     process_and_save_activation_data(input_path, output_file)


def analyze_saved_activation_data(input_file, start_percentile, end_percentile):
    """
    Load processed activation data and find the interval for a given percentile range.

    :param input_file: Path to the JSON file with processed data
    :param start_percentile: The start of the percentile range (0-100)
    :param end_percentile: The end of the percentile range (0-100)
    :return: A dictionary containing analysis results and the requested interval
    """
    # Load the data
    with open(input_file, 'r') as f:
        loaded_data = json.load(f)

    df = pd.DataFrame(loaded_data['data'])
    total_count = loaded_data['total_count']

    # Function to find activation interval for a given percentile range
    def find_activation_interval(start_pct, end_pct):
        start_row = df[df['Percentile'] >= start_pct].iloc[0]
        end_row = df[df['Percentile'] >= end_pct].iloc[0]
        
        return {
            'Start_SD': start_row['SD_start'],
            'End_SD': end_row['SD_end'],
            'Start_Activation': start_row['Activation_start'],
            'End_Activation': end_row['Activation_end'],
            'Actual_Start_Percentile': start_row['Percentile'],
            'Actual_End_Percentile': end_row['Percentile']
        }

    # Get the requested interval
    output = find_activation_interval(start_percentile, end_percentile)

    print(output)
    return output



intervals = [(0,25),
             (25,50),
             (50,75),
             (75,90),
             (90,95),
             (95,99),
             (99,99.9),
             (99.9,99.99),
             (99.99,99.999),
             (99.999,99.9999),
             (99.9999,99.99999),
             (99.99999,100)
]



# Initialize a list to store all results
all_results = []

# Analyze data for each layer and interval
for layer in range(12):
    for start, end in intervals:
        output = analyze_saved_activation_data(f'../histograms/blocks.{layer}.mlp.hook_post_cumulative_sum.json', start, end)
        
        # Add layer and interval information to the output
        output['Layer'] = layer
        output['Interval_Start'] = start
        output['Interval_End'] = end
        
        all_results.append(output)

# Create a DataFrame from all results
df_results = pd.DataFrame(all_results)

# Reorder columns for better readability
column_order = ['Layer', 'Interval_Start', 'Interval_End', 'Start_SD', 'End_SD', 
                'Start_Activation', 'End_Activation', 'Actual_Start_Percentile', 'Actual_End_Percentile']
df_results = df_results[column_order]

# Sort the DataFrame by Layer and Interval_Start
df_results = df_results.sort_values(['Layer', 'Interval_Start'])

# Reset index for clean numbering
df_results = df_results.reset_index(drop=True)

# Save the DataFrame to a CSV file
output_file = 'all_layers_intervals_analysis.csv'
df_results.to_csv(output_file, index=False)

print(f"All results have been saved to {output_file}")

# Display the first few rows of the DataFrame
print(df_results.head())