#!/usr/bin/env python

import re
import os

def change_var_value(input_file, var, new_val):
	""" Changes the corresponding value of a variable in a lammps .in file.

	Args:
		input_file (str): path to lammps .in file to be modified from the working directory "jupyter"
		var (str): name of the variable whose value is to be modified
		new_val (float): the new value of var

	Returns:
		NULL
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


def run_sim(file_name, sim_dir, lmp_path):
	""" Run a lammps simulation by executing a lammps .in file.

	Args:
		file_name (str): name (only) of the .in file to be run.
		sim_dir (str): path from working directory of "jupyter" to directory in which the .in file is located.
		lmp_path (str): path to the lmp executable.

	Returns:
		NULL
	"""
	cwd = os.getcwd()
	sim_dir = cwd + '/' + sim_dir
	os.chdir(sim_dir)
	os.system('mpirun -np 4 ' + lmp_path + ' -in ' + file_name)
	os.chdir(cwd)