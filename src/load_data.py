
import os
# os.chdir("D:/OneDrive - Queen's University/UNITS")

import h5py
import numpy as np
import pandas as pd
from aeon import datasets
from scipy.signal import resample

# Function to load variable length data from HDF5
def load_variable_length_data(h5file, dataset_name):
    grp = h5file[dataset_name]
    data = []
    for key in grp.keys():
        data.append(grp[key][:])
    return data

# Load selected data for control
with h5py.File('selected_control_raw_VandP_Pcorrected.h5', 'r') as f:
    control_Vabs_data = load_variable_length_data(f, 'Vabs')
    control_Pabs_data = load_variable_length_data(f, 'Pabs')
    control_Feats_data = load_variable_length_data(f, 'Feats')
    control_headers = load_variable_length_data(f, 'headers')

# Load selected data for stroke
with h5py.File('selected_stroke_raw_VandP_Pcorrected.h5', 'r') as f:
    stroke_Vabs_data = load_variable_length_data(f, 'Vabs')
    stroke_Pabs_data = load_variable_length_data(f, 'Pabs')
    stroke_Feats_data = load_variable_length_data(f, 'Feats')
    stroke_headers = load_variable_length_data(f, 'headers')

# Function to convert lists of ASCII values back to strings
def ascii_to_str(ascii_list):
    return ''.join([chr(c) for c in ascii_list])

# Convert loaded headers back to proper strings
def convert_headers(headers):
    converted_headers = []
    for header in headers:
        converted_header = []
        for field in header:
            if isinstance(field, np.ndarray):
                converted_header.append(ascii_to_str(field))
            else:
                converted_header.append(field)
        converted_headers.append(converted_header)
    return converted_headers

# Function to decode headers
def decode_headers(headers):
    decoded_headers = []
    for header in headers:
        decoded_header = [field.decode('utf-8') if isinstance(field, bytes) else field for field in header]
        decoded_headers.append(decoded_header)
    return decoded_headers

def conver_to_ts (data, y, data_folder, ts_file_name):
    datasets.write_to_tsfile(data, path=data_folder, y=y, problem_name=ts_file_name)



control_headers = pd.DataFrame(decode_headers(convert_headers(control_headers))).T
stroke_headers = pd.DataFrame(decode_headers(convert_headers(stroke_headers))).T

# Get the number of columns in the first DataFrame
num_cols_df1 = control_headers.shape[1]

# Generate new column headers for the second DataFrame
new_columns_df2 = [num_cols_df1 + i + 1 for i in range(stroke_headers.shape[1])]

stroke_headers.columns = new_columns_df2

combined_headers = pd.concat([control_headers, stroke_headers], axis=1)

def return_data(samples):

    a = resample(np.transpose(control_Pabs_data), samples)
    control_Pabs_data_df = pd.DataFrame(a)
    
    b = resample(np.transpose(control_Vabs_data), samples)
    control_Vabs_data_df = pd.DataFrame(b)
    
    c = resample(np.transpose(stroke_Pabs_data), samples)
    stroke_Pabs_data_df = pd.DataFrame(c)
    
    d = resample(np.transpose(stroke_Vabs_data), samples)
    stroke_Vabs_data_df = pd.DataFrame(d)
    
    combined_Pabs_df = pd.DataFrame(np.concatenate((a,c), axis=1))
    combined_Vabs_df = pd.DataFrame(np.concatenate((b,d), axis=1))
    
    return combined_Vabs_df, combined_Pabs_df, control_Pabs_data_df, control_Vabs_data_df, stroke_Pabs_data_df, stroke_Vabs_data_df, combined_headers

# return_data(1152)