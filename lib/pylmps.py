#!/usr/bin/env python

#==================================================#
#| Python module written for lammps-related tasks.|#
#==================================================#


import re
import os
import json
import pandas as pd


def change_var_value(input_file, var, new_val):
	""" Change the value of a variable in a lammps .in file.

	Parameters
	----------
	input_file : str
		Path to lammps .in file that will be modified from the current working directory.

	var : str
		Name of the variable whose value is to be modified.

	new_val : float
		The new value of var.


	Returns
	-------
		None
	"""
	file = open(input_file, 'r')
	lines = file.readlines()
	file.close()

	file = open(input_file, 'w')
	for line in lines:
		cleaned_line = line.rstrip('\n')
		line_split = re.split(r'\s+', cleaned_line)
		if line_split[0] == 'variable':
			variable = line_split[1]
			if variable == var:
				file.write("variable " + var + " equal " + str(new_val) + "\n")
			else:
				file.write(line)
		else:
			file.write(line)
	file.close()


def run_lmp(file_name, sim_dir, lmp_path):
	""" Execute a lammps .in file locally with mpirun enabled.

	A total of 4 cores is assumed for the local CPU.

	Parameters
	----------
	file_name : str
		Name (only) of the .in file to be run.

	sim_dir : str
		(Relative) path from the current working directory to the directory in
		which the .in file is located.
		
	lmp_path : str
		Path to the lmp executable.


	Returns
	-------
		None
	"""
	cwd = os.getcwd()
	sim_dir = cwd + '/' + sim_dir
	os.chdir(sim_dir)
	os.system('mpirun -np 4 ' + lmp_path + ' -in ' + file_name)
	os.chdir(cwd)


def read_json_to_dict(file_path, params=True):
	""" Read a JSON file to Python dictionary.
	
	Parameters
	----------
	file_path : str
		Absolute/relative path to the .json file.
		
	params : bool, optional
		Indicate whether or not the parsed file is params.json.

		If True: the file is structured like a simple Python dict.
		If False: the file is assumed to encode a pandas DataFrame and is most
			likely a product of calling pylmps.str_array_dict_to_json
			or pandas.DataFrame.to_json.
		
		Defaults to True.
	

	Returns
	-------
	dict
		Python dictionary equivalent of the json file with format:
			<variable> (str): <value> (float if params=True or numpy array
			if otherwise)
	"""
	if params:
		with open(file_path, "r") as file:
			output_dict = json.load(file)
	else:
		df = pd.read_json(path_or_buf=file_path)
		output_dict = df.to_dict(orient='list')

	return output_dict


def str_array_dict_to_json(str_arr_dict, to_file_path):
	""" Saves a Python dictionary with the format <str: numpy array> to disk
	as a .json file.
	
	All values (numpy arrays) of the dictionary are assumed to have the same size
	as each array represents one column in the corresponding dataframe encoded by the
	passed-in Python dictionary.
	
	Parameters
	----------
	str_arr_dict : dict 
		Data stored in the form of a Python dictionary with the key-value pairs
		structured as followed:
			<variable/column name> (str): <values/entries> (1-D numpy array)
	
	to_file_path : str
		Absolute/relative path to the file where the Python dictionary will be written
		to disk in JSON format.
		
		Should end with ".json".
		
	
	Returns
	-------
	None
		Output written to file.
	"""
	df = pd.DataFrame.from_dict(str_arr_dict)
	df.to_json(to_file_path)
