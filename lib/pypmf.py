#!/usr/bin/env python

#=====================================================================================#
#| Python module written to study reversible work theorem (potential of mean force). |#
#=====================================================================================#


import numpy as np
import re
import os
import matplotlib.pyplot as plt
import math
import glob
import sys
#!!!!!---------------------------------------------------------------------!!!!!#
#! Path to `lib` directory may change depending on its actual location on disk !#
#!!!!!---------------------------------------------------------------------!!!!!#
sys.path.append('../lib')
import pylmps


def apply_rwt_to_rdf(num_bin, cutoff, rdf, temp):
	""" Apply the reversible work theorem to a radial distribution function
	to calculate the derivative of the potential of mean force (i.e. the
	force distribution).

	Warnings are automatically ignored before and re-enabled after this function
	is executed. Read source code for more detail.

	Parameters
	----------
	num_bin : int
		The number of bins used for the RDF.

	cutoff : int
		Maximum `r` at which the RDF is calculated.

	rdf : 1-D numpy array
		RDF, g(r), at each bin spanning across `r` until r=`cutoff`.

	temp : float
		The temperature of the system of which the `rdf` is computed.


	Returns
	-------
	A 1-D numpy array:
		The parallel force distribution as calculated using the reversible work
		theorem.
	"""
	bin_width = cutoff/num_bin

	# divide-by-0 warning would be raised by numpy.log
	# overflow warning would be raised by numpy.gradient
	np.seterr(divide='ignore', over='ignore')
	
	log_rdf = np.nan_to_num(np.log(rdf))
	f = np.gradient(log_rdf, bin_width) * temp

	np.seterr(divide='warn', over='warn')

	return f


def check_rwt(num_bin, cutoff, rdf, force, temp, **kwargs):
	""" Check if the reversible work theorem is valid for a given system/setup
	by plotting.

	All positional arguments (`num_bin`, `cutoff`, `rdf`, `force, and `temp`)
	should describe the same system/setup.

	It is assumed that both `rdf` and `force` have the same `num_bin` and
	`cutoff`.

	Parameters
	----------
	num_bin : int
		The number of bins used for the RDF and (parallel) force distribution.

	cutoff : int
		Maximum `r` at which the RDF and force distribution are calculated.

	rdf : 1-D numpy array
		RDF, g(r), at each bin spanning across `r` until r=`cutoff`.

	force : 1-D numpy array
		Interparticulate force (i.e. force distribution) at each bin, spanning
		across `r` until r=`cutoff`.

		Typically as returned by `read_force_distribution`.

	temp : float
		The temperature of the system of study.

	**kwargs : dict
		Keyword arguments concerning the final figure.
		
		title : str
			Title of plot.

		xlim : 2-element list/array
			Limits for the x-axis of the plot in the form [lowerbound, upperbound]

		ylim : 2-element list/array
			Limits for the y-axis of the plot in the form [lowerbound, upperbound]

		save_to : str
			Path to the location at which the generated plot will be written
			to disk.

			Should end with '.png'


	Returns
	-------
	None.
		An overlaid lineplot is output when function is called in jupyter
		notebook environment.

		The same plot will also be written to disk if `save_to` is passed in.
	"""
	force_rwt = apply_rwt_to_rdf(num_bin, cutoff, rdf, temp)

	x_ax = np.linspace(0, cutoff, num_bin)

	plt.figure()
	plt.plot(x_ax, force)
	plt.plot(x_ax, force_rwt)
	plt.legend([r'PMF: $\left<\frac{d}{dr_{1}}U\right>_{r_{1},r_{2}}$',
				r'RDF: $k_{B}T\frac{d}{dr_{1}}(ln\hspace{0.2}g(r_{1},r_{2}))$'])
	plt.xlabel('$|r_{1}-r_{2}|$')
	plt.axhline(y = 0, ls = '--', c = 'k')

	if 'xlim' in kwargs:
		xlim = kwargs['xlim']
		plt.xlim(xlim[0], xlim[1])
	if 'ylim' in kwargs:
		ylim = kwargs['ylim']
		plt.ylim(ylim[0], ylim[1])
	if 'title' in kwargs:
		plot_title = kwargs['title']
		plt.title(plot_title)
	if 'save_to' in kwargs:
		save_path = kwargs['save_to']
		plt.savefig(save_path, dpi=400, bbox_inches='tight')


def read_multiple_temps(average, *args):
	""" Read system temperature from multiple files.

	Targeted files should be properly formatted, specifically they each should be
	an output from calling compute temp in lammps.

	Parameters
	----------
	average : bool
		Whether or not an average of all the temperatures will be taken.

	*args : iterable of str
		Paths to files each containing temperature data.


	Returns
	-------
	If `average`=True:
		A float that is an average temperature.

	otherwise:
		A 1-D numpy array of temperatures, each corresponds to one file, in order.
	"""
	first_pass = True

	for path in args:
		curr_temp = read_temp(path)

		if first_pass:
			all_temps = np.array([curr_temp])
			first_pass = False
		else:
			all_temps = np.append(all_temps, curr_temp)

	if average:
		return np.mean(all_temps)
	return all_temps


def read_temp(temp_file_path):
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


def read_force_distribution(job_dir, *args):
	""" Read parallel force (force along the axis connecting centers of two particles)
	distribution from all JSON files in a given directory.

	Handles parsing multiple files; in this case, the distributions are combined
	by taking an average properly.

	This function assumes that the .json files and given directories are properly
	formatted as a result of executing py_submit_serial.sh and force_distribution.py
	on nersc and that ONLY concerned .json files are in the directory.

	Parameters
	----------
	job_dir : str
		Path to a '.../post_proc/jobs/' directory that has subdirectories named according
		to each submitted post-processing job on nersc.

		Should end with '/' (or commonly 'jobs/')

	*args : iterable of str
		Slurm ID of job(s) that post-processed position-force data to compute
		parallel force distribution.


	Returns
	-------
	A 1-D numpy array:
		Parallel force distribution as a function of distance between particles.
	"""
	parallel_dist = []

	for j in args:
		force_files = job_dir + j + '/*.json'
		for f in glob.iglob(force_files):
			force_dict = pylmps.read_json_to_dict(f, params=False)
			parallel_dist.append(force_dict['parallel forces'])

	return np.mean(np.array(parallel_dist), axis=0)

	
def calc_force_distribution(file_path, num_bin, cutoff, lx, ly, mol=False):
	""" Calculates the effective force decompositions between any pair of particles/atoms based on their position and force data and
	averages the forces according to pre-defined bins of interparticular distance.
	
	Force decompositions are in the interparticle/parallel and transverse directions. Sign convention is: 1) a positive interparticle
	force is effective "repulsion" and 2) a positive transverse force represents effective "rotation" in the
	counterclockwise direction.

	This function is capable of both reading and processing the position and force data.


	Args:
		file_path (string): RELATIVE path to the file containing position-force data.
		num_bin (int): desired number of bins for the distributions of pair-wise force w.r.t. inter-particle distance.
		cutoff (float): upperbound (in appropriate units) of the force distribution (i.e. of the last bin).
		lx (float): length of the simulation box in x direction.
		ly (float): length of the simulation box in y direction.
		mol (bool): whether or not the particles (whose position and force data are recorded) belong to molecules (as opposed
		to being individual particles themselves). Default to False.

	Returns:
		dict: distribution of force decompositions and their corresponding bins:
			"bins": 1-D numpy array with each entry denoting lower/upper bound for each bin.
			"parallel forces": 1-D numpy array representing the average parallel force between any particle pair in each bin.
			"transverse forces": 1-D numpy array representing the average transverse force between any particle pair in each bin.

	"""
	paral_dict = {}
	trans_dict = {}

	data_arr = read_pos_force(file_path)
	for i in range(len(data_arr)):
		one_bins = bin_by_dist(num_bin, cutoff, data_arr[i], lx, ly, mol)
		for key in one_bins.keys():
			vals = one_bins[key]
			for force_pair in vals:
				if key not in paral_dict:
					paral_dict[key] = [force_pair[0]]
				else:
					temp = paral_dict[key]
					temp.append(force_pair[0])
					paral_dict[key] = temp
				if key not in trans_dict:
					trans_dict[key] = [force_pair[1]]
				else:
					temp = trans_dict[key]
					temp.append(force_pair[1])
					trans_dict[key] = temp

	paral_arr = np.zeros(num_bin)
	trans_arr = np.zeros(num_bin)

	for bin_label in range(num_bin):
		if bin_label in paral_dict:
			arrp = paral_dict[bin_label]
			fp_avg = np.average(arrp)
			paral_arr[bin_label] = fp_avg
		if bin_label in trans_dict:
			arrt = trans_dict[bin_label]
			ft_avg = np.average(arrt)
			trans_arr[bin_label] = ft_avg

	bins = np.linspace(0, cutoff, num_bin)

	force_dict = {}
	force_dict["bins"] = bins
	force_dict["parallel forces"] = paral_arr
	force_dict["transverse forces"] = trans_arr

	return force_dict


def read_pos_force(file_path):
	""" Read data on the positions of and forces on passive particles in a simulated active dumbbell system. Format
	of the file to be read is strictly assumed to follow that generated by LAMMPS' 'dump custom' command.

	@param FILE_PATH: A string representing the path from the CWD of this notebook to the file to be parsed.

	@return: An appropriate 3-D array with dimensions 1, 2, 3 delineating each timestep, particle, and data
	parameter [(mol), x, y, fx, fy], respectively
	"""
	final_arr = []
	atom_lst = []

	file = open(file_path, 'r')
	next_line = file.readline()
	while next_line != '':
		if next_line == 'ITEM: TIMESTEP\n':
			if atom_lst:
				final_arr.append(atom_lst)
				atom_lst = []
			for i in range(8):
				file.readline()
			next_line = file.readline()
		else:
			data = next_line.rstrip('\n').rstrip()
			data_split = re.split(r'\s', data)
			data_float = []
			for s in data_split:
				data_float.append(float(s))
			atom_lst.append(data_float)
			next_line = file.readline()
	if atom_lst:
		final_arr.append(atom_lst)
	file.close()

	return final_arr


def bin_by_dist(num_bin, cutoff, arr, lx, ly, mol):
	""" Bin each possible passive-particle pair given by ARR according to its minimum-image-convention Euclidean
	distance. Pair parameters such as interparticle and transverse forces are calculated.

	@param NUM_BIN: An integer value denoting the desired number of bins.
	@param CUTOFF: A numerical value denoting the upperbound (in appropriate units) of the histogram/of the last bin.
	@param ARR: A 2-D array with dimension 1 separating each passive particle and 2 separating each data parameter in
	the format of [x, y, fx, fy]. This array should correspond to the second and third dimensions of the 3-D array
	returned by the function READ_POS_FORCE.
	@param LX: Length of the simulation box in x direction.
	@param LY: Length of the simulation box in y direction.
	@param MOL: A boolean indicating whether or not the subject of the calculated distribution are molecules (as opposed to
		individual atoms/particles), which affects output lammps file format and force calculations.

	@return: A dictionary with the keys representing the integer bin label (from 0 to NUM_BIN-1) and the corresponding
	values being 2-D arrays with dimension 1 separating each pair in that particular bin and dimension 2 being
	[interparticle, transverse] forces.
	"""
	bins = {}
	bin_width = cutoff / num_bin

	for index1 in range(len(arr) - 1):
		p1 = arr[index1]
		if mol:
			mol_id1 = p1[0]
			p1 = p1[1:]
		for index2 in range(index1 + 1, len(arr)):
			p2 = arr[index2]
			if mol:
				mol_id2 = p2[0]
				p2 = p2[1:]
			if not mol or (mol and mol_id1 != mol_id2):
				dist_data = calc_min_dist(p1, p2, lx, ly)
				dist = dist_data[0]
				if dist <= cutoff:
					bin_label = int(dist // bin_width)
					dx_min = dist_data[1]
					dy_min = dist_data[2]
					forces = calc_forces(p1, p2, dx_min, dy_min)
					if bin_label in bins:
						temp = bins[bin_label]
						temp.append(forces)
						bins[bin_label] = temp
					else:
						bins[bin_label] = [forces]

	return bins


def calc_min_dist(p1, p2, lx, ly):
	""" Calculate the Euclidean distance between two particles using minimum image convention.

	@param P1: An array with format [x, y, fx, fy] for the first particle.
	@param P2: An array with format [x, y, fx, fy] for the second particle.
	@param LX: Length of the simulation box in x direction.
	@param LY: Length of the simulation box in y direction.

	@return: An array [dist_min, dx_min, dy_min] between the two particles (in appropriate units).

	Reference: ResearchGate "Efficient Coding of the Minimum Image Convention"
	"""
	x1 = p1[0]
	x2 = p2[0]
	y1 = p1[1]
	y2 = p2[1]
	dx = x1 - x2
	dy = y1 - y2
	lx2 = 0.5 * lx
	ly2 = 0.5 * ly
	kx = int(dx / lx2)
	ky = int(dy / ly2)
	dx_min = dx - kx * lx
	dy_min = dy - ky * ly
	dist_min = ((dx_min**2) + (dy_min**2))**0.5

	return [dist_min, dx_min, dy_min]


def calc_forces(p1, p2, dx_min, dy_min):
	""" Calculate the interparticle and transverse forces between a pair of particles. The convention used for the
	forces in each direction is: 1) a positive interparticle force is effective "repulsion" and 2) a positive
	transverse force represents effective "rotation" in the counterclockwise direction.

	@param P1: An array with format [x, y, fx, fy] for the first particle.
	@param P2: An array with format [x, y, fx, fy] for the second particle.
	@param DX_MIN: Calculated distance in the x direction between the two particles using minimum image convention.
	@param DY_MIN: Calculated distance in the y direction between the two particles using minimum image convention.

	@return: An array in the form of [interparticle, transverse] forces.

	Reference: Wikipedia "Rotation of axes"
	"""
	fx1 = p1[2]
	fy1 = p1[3]
	fx2 = p2[2]
	fy2 = p2[3]

	# special case: when the two particles are perfectly vertically aligned
	if dx_min == 0:
		if dy_min > 0:
			finter = fy1 - fy2
			ftrans = -fx1 + fx2
		else:
			finter = -fy1 + fy2
			ftrans = fx1 - fx2
		return [finter, ftrans]

	# general cases
	angle_abs = math.atan(abs(dy_min) / abs(dx_min))
	if dx_min > 0:
		if dy_min >= 0:
			angle1 = angle_abs
			angle2 = angle_abs + math.pi
		else:
			angle1 = -angle_abs
			angle2 = -angle_abs - math.pi
	elif dx_min < 0:
		if dy_min >= 0:
			angle1 = math.pi - angle_abs
			angle2 = -angle_abs
		else:
			angle1 = angle_abs + math.pi
			angle2 = angle_abs

	# rotate axes accordingly to calculate finter and ftrans for each particle
	finter1 = fx1 * math.cos(angle1) + fy1 * math.sin(angle1)
	ftrans1 = -fx1 * math.sin(angle1) + fy1 * math.cos(angle1)

	finter2 = fx2 * math.cos(angle2) + fy2 * math.sin(angle2)
	ftrans2 = -fx2 * math.sin(angle2) + fy2 * math.cos(angle2)

	# add the forces to find total finter and ftrans
	finter = (finter1 + finter2) / 2
	ftrans = (ftrans1 + ftrans2) / 2

	return [finter, ftrans]


def compare_rdf_pmf(num_bin, cutoff, pmf_distr, parallelized=True, savefig=True, **kwargs):
	""" @deprecated
	Compare graphically a PMF distribution generated through simulation and post-processing with one that is generated
	through applying the derivative form of reversible work theorem on a RDF; the RWT calculation is performed within this
	function.

	Parameters
	----------
		num_bin (int): the desired number of bins for either distribution.
		cutoff (float): the upperbound (in appropriate units) of the histogram/of the last bin.
		pmf_distr (1-D array): the histogram/distribution of any type of force (parallel or tranverse) that
			has num_bin many bins.
		savefig (bool): determines whether or not the generated plot is saved to disk. Default to True.
		
	Returns
	-------
		NULL: the comparison plot is exported into the directory fig_dir and nothing is returned.
		
		
		
	for **kwargs:
		title (str): title label of the generated plot.
		y_lim (1-D array): formatted as the desired [lowerbound, upperbound] of the y-axis of the plot.
		
		IF SAVEFIG = TRUE:
			png_name (str): name of the .png file containing the exported/saved plot.
			"fig_dir": directory in which figure generated by this function should be saved.
		IF PARALLELIZED = FALSE:
			paths (str-str dict): the paths to files/directories from the working directory of "jupyter".
			The paths should consist of:
				"rdf_path": file containing rdf data output from LAMMPS
				"temp_path": file with averaged temperature of simulation bath
		ELSE:
			job_dir: lammps job directory with parallelized runs (e.g. 1-30) ends with '/'
	"""
	if parallelized:
		job_dir = kwargs['job_dir']
		rdf_path = job_dir + '1/data/cumul_rdf.rdf'
		temp_path = job_dir + '1/data/passive_temp.temp'

		_, rdfs = read_rdf_arrays(rdf_path, cumul=True)
		temps = np.array([read_passive_temp(temp_path)])

		for i in np.arange(2, 31):
			rdf_path = job_dir + str(i) + '/data/cumul_rdf.rdf'
			temp_path = job_dir + str(i) + '/data/passive_temp.temp'

			_, curr_rdf = read_rdf_arrays(rdf_path, cumul=True)
			rdfs = np.vstack((rdfs, curr_rdf))
			temps = np.append(temps, read_passive_temp(temp_path))

		rdf = np.mean(rdfs, axis=0)
		T = np.mean(temps)
	else:
		rdf_path = kwargs['rdf_path']
		temp_path = kwargs['temp_path']

		_, rdf = read_rdf_arrays(rdf_path)
		T = read_passive_temp(temp_path)

	
	bin_width = cutoff/num_bin
	
	# reversible work theorem (RWT)
	pmf_rwt = np.gradient(np.log(rdf), bin_width) * T
	
	x_ax = np.linspace(0, cutoff, num_bin)

	plt.figure()
	plt.plot(x_ax, pmf_distr)
	plt.plot(x_ax, pmf_rwt)
	if 'xlim' in kwargs:
		xlim = kwargs['xlim']
		plt.xlim(xlim[0], xlim[1])
	if 'ylim' in kwargs:
		ylim = kwargs['ylim']
		plt.ylim(ylim[0], ylim[1])
	if 'title' in kwargs:
		plot_title = kwargs['title']
		plt.title(plot_title)
	plt.xlabel('$|r_{1}-r_{2}|$')
	plt.legend([r'PMF: $\left<\frac{d}{dr_{1}}U\right>_{r_{1},r_{2}}$',
				r'RDF: $k_{B}T\frac{d}{dr_{1}}(ln\hspace{0.2}g(r_{1},r_{2}))$'])
	plt.axhline(y = 0, ls = '--', c = 'k')
	if savefig:
		fig_dir = kwargs['fig_dir']
		png_name = kwargs['png_name']
		plt.savefig(fig_dir + '/' + png_name + '.png', dpi=400, bbox_inches='tight')
