import pandas as pd
import numpy as np

import json
from tqdm.auto import tqdm




def analyze_saved_activation_data(input_file, start_percentile, end_percentile):
    """
    Load processed activation data and find the interval for a given percentile range.

    :param input_file: Path to the JSON file with processed data
    :param start_percentile: The start of the percentile range (0-100)
    :param end_percentile: The end of the percentile range (0-100)
    :return: A dictionary containing analysis results and the requested interval
    """
    df = pd.read_csv(input_file, sep='\t')
    
    start_row = df[df['Percentile'] >= start_pct].iloc[0]
    end_row = df[df['Percentile'] >= end_pct].iloc[0]

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
        output = analyze_saved_activation_data(f'../histograms/mlp.hook_out/blocks.{layer}.hook_mlp_out_percentiles.txt', start, end)
        
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
output_file = 'all_layers_intervals_analysis_mlp_out.csv'
df_results.to_csv(output_file, index=False)

print(f"All results have been saved to {output_file}")

# Display the first few rows of the DataFrame
print(df_results.head())