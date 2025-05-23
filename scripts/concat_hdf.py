# SPDX-FileCopyrightText: 2025 University of Manchester
#
# SPDX-License-Identifier: apache-2.0

import os
from src.data import concat_user_hdf_files


def get_hdf_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter the list to include only hdf files
    hdf_files = [file for file in files if file.endswith('.hdf')]
    
    return hdf_files


directory = "data/ex1/from_julia/"
file_list = get_hdf_files(directory)
file_list = [directory + file for file in file_list]
                          
output_file = "data/ex1/from_julia/concat/concat.hdf"
concat_user_hdf_files(file_list,output_file)