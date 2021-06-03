#!/usr/bin/env python

#=================================================================#
#| Python module written to study radial distribution functions. |#
#=================================================================#


import numpy as np
import re as re
import os as os
import matplotlib.pyplot as plt


def count_total_lines(file_path):
	""" Count total number of lines in the file.

	Parameters
	----------
	file_path : str
		Path to the file to be counted.


	Returns
	-------
	An integer:
		Total number of lines in the file.
	"""
	file = open(file_path, 'r')
	line_count = 0
	parsed_line = file.readline()
	while (parsed_line != ''):
		line_count += 1
		parsed_line = file.readline()
	file.close()
	return line_count


def read_rdf_arrays(file_path, num_bin=100, cumul=True):
	""" Read RDF data from file into numpy arrays.
	
	It is assumed that there are four columns in file_path, as properly formatted by
	calling compute rdf in lammps.

	Can handle multiple RDFs all with the same number of bins stored in the same
	file.

	Parameters:
	-----------
	file_path : str
		Path to file that contains RDF data.

	num_bin : int, optional
		Number of bins used to compute the parsed RDF.

		Default to 100.

	cumul : bool, optional
		Indicate whether or not the parsed RDF is cumulative (i.e. if the file
		contains only one RDF vs. many RDFs scanning across time.)


	Returns
	-------
	A tuple of 2 numpy arrays:
		1) 1-D array that represents r values (x-axis values of a RDF plot) 
			that correspond to the center of each bin in the distribution.

		2)
		If cumul=True:
			1-D array storing the radial density, g(r), of a RDF.
		
		otherwise:
			2-D array with each scan of the radial density separated by the zeroth-axis.
			In other words, the 1-D array stored at each of the position along
			the 0th-axis corresponds to one complete radial density.
	"""

	file = open(file_path, 'r')

	r_lst = []
	rdf_lst = []
	if not cumul:
		one_rdf_scan = []

	curr_ln = file.readline()
	start_read = False
	first_pass = True
	curr_bin = 1

	while curr_ln != '':
		# process current line of text
		curr_ln = curr_ln.rstrip('\n')
		curr_ln_split = re.split(r'\s', curr_ln)
		
		# first line of data reached
		if len(curr_ln_split) == 4:
			start_read = True

		if start_read:
			# if the rdf data is cumulative with one scan only
			if cumul:
				r_lst.append(float(curr_ln_split[1]))
				rdf_lst.append(float(curr_ln_split[2]))
			# if the rdf data consist of multiple time-averaged scans
			else:
				# reset variables for next time scan
				if curr_bin > num_bin:
					rdf_lst.append(one_rdf_scan)
					curr_ln_split = re.split(r'\s', file.readline().rstrip('\n'))
					one_rdf_scan = []
					curr_bin = 1
					first_pass = False
				
				one_rdf_scan.append(float(curr_ln_split[2]))
				curr_bin += 1
				
				if first_pass:
					r_lst.append(float(curr_ln_split[1]))

		curr_ln = file.readline()
	
	if not cumul:
		rdf_lst.append(one_rdf_scan)
	file.close()

	r_arr = np.array(r_lst)
	rdf_arr = np.array(rdf_lst)

	return r_arr, rdf_arr


def read_passive_temp(temp_file_path):
	""" Read system temperature from file.

	Targeted file should be properly formatted, specifically it should be an
	output from calling compute temp in lammps.

	Parameters
	----------
	temp_file_path : str
		Path to file with system temperature printout.


	Returns
	-------
	A float:
		The system temperature.
	"""
	file = open(temp_file_path, 'r')
	file_rows = file.readlines()
	temp_row = file_rows[2]
	temp_row = temp_row.rstrip('\n')
	row_split = re.split(r'\s', temp_row)
	temp = float(row_split[1])
	file.close()

	return temp


def plot_rdf_convergence(Nfreq, at_bin, rdfs, refline_shift=2, **kwargs):
	""" Visualize (approximately) the convergence of radial distribution function using
	the Central Limit Theorem.

	Parameters
	----------
	Nfreq : int
		The frequency (per number-of-timesteps basis) at which RDF is written to file.

	at_bin : int
		The bin at which rdf convergence is investigated.

	rdfs : 2-D numpy array
		Scan of RDF across simulation time.

		Should be (or match in format with) an output of read_rdf_arrays
		when cumul=False.

	refline_shift : float
		Hyperparameter that shifts the reference line up (larger value) or down
		(smaller value).

		Default to 2.
	
	**kwargs :
		Additional arguments affecting visualization.

		xlim : tuple
			x-axis bounds of the plot.

		ylim : tuple
			y-axis bounds of the plot.


	Returns
	-------
	Plot in output cell.
	"""
	scans, running_abs_error = calc_running_abs_error(at_bin, rdfs)
	
	n = Nfreq*scans
	refline = (10**refline_shift) * np.sqrt(1./n)
	
	plt.figure()
	plt.plot(n, running_abs_error, label=r"$\epsilon(n)$")
	plt.plot(n, refline, label=r"$\propto 1/\sqrt{n}$")
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel("Number of Timesteps (n)")
	plt.ylabel(r"Absolute Error $\epsilon(n)$")
	plt.title("Running Absolute Error of RDF at Bin {}".format(at_bin))
	plt.legend()
	if 'xlim' in kwargs:
		xlim = kwargs['xlim']
		plt.xlim(xlim)
	if 'ylim' in kwargs:
		ylim = kwargs['ylim']
		plt.ylim(ylim)


def calc_running_abs_error(at_bin, rdfs):
	""" Calculate the running error of RDF over simulation time at a
	specified bin.

	Uses the g(r) averaged over all samples as the reference/true value when
	the absolute error is calculated.

	Parameters
	----------
	at_bin : int
		The bin at which the error of RDF is calculated.

	rdfs : 2-D numpy array
		Scan of RDF across simulation time.

		Should be (or match in format with) an output of read_rdf_arrays
		when cumul=False.

		Usually passed into this function from a call in plot_rdf_convergence.


	Returns
	-------
	2 1-D numpy arrays:
		1) an integer sequence running from 1 to the length of the zeroth
			dimension of rdfs.

		2) running absolute errors with the same number of entries as the
			first array output.
	"""
	if at_bin > rdfs.shape[1]:
		print("Invalid bin, exceeds number of bins used.")
		return None
	
	idx = at_bin - 1
	rdf_at_bin = rdfs[:, idx]
	num_scans = len(rdf_at_bin)
	scans = np.arange(1, num_scans+1)
	
	best_estimate = np.mean(rdf_at_bin)
	running_avg = np.cumsum(rdf_at_bin) / scans
	running_abs_error = np.abs(running_avg - best_estimate)
	
	return scans, running_abs_error


def make_overlaid_plots(paths, labels, title, save_path=''):
	""" Overlay multiple rdf curves in one properly formatted plot.

	Generated plot can be saved to disk.

	Args:
		paths (iterable of strings): paths from the current notebook directory to the .rdf files to be parsed in order.
		labels (iterable of strings): legend labels for each rdf curve whose path is listed in paths; should be ordered
		in correspondence and equal in length to paths.
		title (string): title of the rdf plot.
		save_path (string): path indicating the file to which the generated figure should be saved on disk; no figure will
		be saved if no path is entered (i.e. the default, an empty string). Should end in '.png'

	Returns:
		NULL. The rdf plot will be output to screen, or even saved to disk.
	"""
	plt.figure()
	plt.xlabel("r")
	plt.ylabel("g(r)")
	plt.title(title)
	plt.axhline(y=1, ls='--', c='k')

	for i in range(len(paths)):
		file_path = paths[i]
		label = labels[i]
		r_arr, rdf_arr = read_rdf_arrays(file_path)
		plt.plot(r_arr, rdf_arr, label = label)

	plt.legend()

	if save_path != '':
		plt.savefig(save_path, dpi=400, bbox_inches='tight')


def make_rdf_from_arr(rdf_arr, plot_title, label=""):
	""" Generate a RDF plot using RDF data passed in as an argument. This function (so far) is designed
	solely for data visualization (i.e. resulting figures will not be exported as .png high quality images)

	@param RDF_ARR: A 2-D array consisted of 4 columns of RDF data; This array should closely
		follow the format of the array generated by c_rdf_file_to_array.
	@param PLOT_TITLE: A string specifying the title of the RDF plot.
	@param LABEL: A string encoding the legend label that the RDF plot should be labeled with; defaults to no label.

	@return A single RDF line plot generated using RDF_ARR with title PLOT_TITLE.
	"""
	# Extract compute_rdf parameters
	num_bin = len(rdf_arr)
	bin_width = rdf_arr[0][1] * 2

	# rdf_cutoff = num_bin * bin_width    # cutoff distance for c_rdf

	# Plot rdf as a line plot
	rdf_r = np.empty(num_bin)
	rdf_density = np.empty(num_bin)

	for i in range(num_bin):
		rdf_r[i] = rdf_arr[i][1]
		rdf_density[i] = rdf_arr[i][2]

	if label != "":
		plt.plot(rdf_r, rdf_density, label=label)
		plt.legend()
	else:
		plt.plot(rdf_r, rdf_density)
	plt.xlabel("r")
	plt.ylabel("g(r)")
	plt.title(plot_title)
	plt.axhline(y=1, ls='--', c='k')
	plt.show()


def write_force_temp_pair(force_temp_dict, force_temp_file_path):
	""" Write dictionary <active force magnitude> : <system temperature> to file.

	Note: The final file on disk does not follow JSON format, but rather should
	simply be of .txt format.

	Parameters
	----------
	force_temp_dict : dict
		The temperature of the system for different magnitudes of active force.

	force_temp_file_path : str
		Path to file in which force_temp_dict is stored.


	Returns
	-------
	None:
		Data written to file on disk.
	"""
	file = open(force_temp_file_path, 'w')
	keys = force_temp_dict.keys()
	for key in keys:
		value = force_temp_dict[key]
		line = str(key) + " " + str(value) + "\n"
		file.write(line)
	file.close()


def c_rdf_file_to_array(file_path):
	""" @deprecated, but used by other functions in `pyrdf` module. Use
	`read_rdf_arrays` instead.

	Read RDF data from .rdf file into a numpy array. It is assumed that there are four columns
	in the .rdf file, that the same file records time-averaged, cumulative RDF data, and that data
	entries start on Line 5 of the file.
	@param FILE_PATH: A string representing the path from the CWD of this notebook to the .rdf
		file to be parsed.
	@return A 2-dimensional array with rows representing rdf bins and columns representing bin label,
		bin center (r) value, radial distribution g(r), and coordination number, respectively.
	"""
	num_lines = count_total_lines(file_path)

	file = open(file_path, 'r')

	num_rows = num_lines - 4
	arr = np.empty([num_rows, 4])

	for i in range(1, 5):
		file.readline()

	for i in range(5, num_lines + 1):
		row_str = file.readline()
		row_str = row_str.rstrip('\n')
		row_str_split = re.split(r'\s', row_str)
		for j in range(4):
			arr[i - 5][j] = float(row_str_split[j])

	file.close()
	return arr


def plot_mult_rdf(var, var_values, sim_file_name, rdf_file_path, extra_args=[]):
	""" @deprecated

	Generate RDF calculations using LAMMPS by varying a specified variable over a specified
	value set. This is an encapsulating function that will edit the variable value in the LAMMPS input
	file, run the simulation, read in RDF data, and plot the overlaid RDF, while finally exporting the
	subsequent plot as a .png image to ../examples/active/figures directory. Extra functionalities are
	available.

	@param VAR: A string denoting the simulation parameter to be changed between simulation runs.
	@param VAR_VALUES: A tuple/array specifying the set of values VAR should take between each simulation run.
	@param SIM_FILE_NAME: A string representing the name of the LAMMPS .in file that needs to be modified;
		should be in the format <file_name>.in
	@param RDF_FILE_PATH: A string representing the path from this notebook CWD to the LAMMPS .rdf file that
		is output from each simulation run and containing RDF calculation data used to plot radial distributions.
	##########################################################################################################
	@param EXTRA_ARGS: An array containing any extra argument that is needed to employ the extended functionality
		of this function. Defaults to an empty array, denoting that only basic functionality is used. The extra
		functionalities include:
			1) obtain temperatures of the passive WCA particles computed during equilibrated runs and write them
			to a designated file along with the corresponding f_active's. Arguments include: Use simultaneously
			with f_active.........
			[boolean, lammp_output_path, wca_temp_file_path]
	##########################################################################################################

	@return Multiple RDF line plots generated using RDF data obtained for each simulation run.
	"""
	# declare variables for extra functionalities
	tmps = []

	# pick out variables that are kept constant between runs
	full_param_lst = ["rhoactive", "rhopassive", "gamma", "f_active"]
	param_lst = []
	for p in full_param_lst:
		if p != var:
			param_lst = param_lst + [p]

	cwd = os.getcwd()
	# directory in which LAMMPS .in file is located
	sim_dir = cwd + '/../examples/active'

	params = {}
	for curr_val in var_values:
		# rewrite the .in file with new VAR value
		sim_file_path = "../examples/active/" + sim_file_name
		sim_file = open(sim_file_path, 'r')
		lines = sim_file.readlines()
		sim_file.close()
		sim_file = open(sim_file_path, 'w')
		for line in lines:
			cleaned_line = line.rstrip('\n')
			line_split = re.split(r'\s+', cleaned_line)
			if line_split[0] == 'variable':
				variable = line_split[1]
				if variable == var:
					sim_file.write("variable " + var +
								   " equal " + str(curr_val) + "\n")
				else:
					if variable in param_lst:
						params[variable] = line_split[3]
					sim_file.write(line)
			else:
				sim_file.write(line)
		sim_file.close()

		# run simulation
		os.chdir(sim_dir)
		os.system('mpirun -np 4 ~/Desktop/lammps/build/lmp -in ' + sim_file_name)
		os.chdir(cwd)

		# execute extra functionalities BETWEEN runs
		if extra_args:
			lmp_temp_out_file = extra_args[0][1]
			temp = read_passive_temp(lmp_temp_out_file)
			tmps = tmps + [temp]

		# plot rdf of the current simulation run
		total_line_count = count_total_lines(rdf_file_path)
		rdf_arr = c_rdf_file_to_array(rdf_file_path)
		num_bin = len(rdf_arr)
		bin_width = rdf_arr[0][1] * 2
		rdf_r = np.empty(num_bin)
		rdf_density = np.empty(num_bin)

		for i in range(num_bin):
			rdf_r[i] = rdf_arr[i][1]
			rdf_density[i] = rdf_arr[i][2]

		plt.plot(rdf_r, rdf_density, label=var + ": " + str(curr_val))

	# execute extra functionalities AFTER runs
	if extra_args:
		wca_temp_input_file = extra_args[0][2]
		force_temp = {}
		for i in range(len(var_values)):
			force_temp[var_values[i]] = tmps[i]
		write_force_temp_pair(force_temp, wca_temp_input_file)

	# add plot features and export plot
	var_vals = ""
	for parameter in param_lst:
		var_vals += parameter + "=" + params[parameter] + " "
	var_vals = var_vals.rstrip()
	plot_title = "RDF of Active Dumbbells with " + var_vals
	plt.title(plot_title)
	plt.xlabel("r")
	plt.ylabel("g(r)")
	plt.legend()
	plt.axhline(y=1, ls='--', c='k')
	# plt.savefig("../examples/active/figures/" + var_vals +
	#             ".png", dpi=400, bbox_inches='tight')
	plt.show()
