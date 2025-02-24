"""
EDF Reader of EyeLink Data

Adapted from: https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/edfreader.py
Original Author: Edwin Dalmaijer
License: GPU GPL v3 

Adapted By: Han Zhang <hanzh@umich.edu>
Date: 12/25/2024

Changes:
- Added support for reading metadata.
- Added support for storing the last message and its time for each sample and event.
- Moved checking trial end to the end of the loop to allow the last line (stop MSG) to be extracted.
"""

__author__ = "Edwin Dalmaijer, Han Zhang"
__credits__ = ["Edwin Dalmaijer"]
__license__ = "GNU GPL v3"
__maintainer__ = "Han Zhang"
__email__ = "hanzh@umich.edu"
__status__ = "Adapted"



import copy
import os.path
import numpy
from tqdm.notebook import tqdm
import re

def replace_missing(value, missing=0.0):
	
	"""Returns missing code if passed value is missing, or the passed value
	if it is not missing; a missing value in the EDF contains only a
	period, no numbers; NOTE: this function is for gaze position values
	only, NOT for pupil size, as missing pupil size data is coded '0.0'
	
	arguments
	value		-	either an X or a Y gaze position value (NOT pupil
					size! This is coded '0.0')
	
	keyword arguments
	missing		-	the missing code to replace missing data with
					(default = 0.0)
	
	returns
	value		-	either a missing code, or a float value of the
					gaze position
	"""
	
	if value.replace(' ','') == '.':
		return missing
	else:
		return float(value)

def read_edf(filename, start, stop=None, missing=0.0, debug=False):
	
	"""Returns a list with dicts for every trial. A trial dict contains the
	following keys:
		x		-	numpy array of x positions
		y		-	numpy array of y positions
		size		-	numpy array of pupil size
		time		-	numpy array of timestamps, t=0 at trialstart
		trackertime	-	numpy array of timestamps, according to EDF
		events	-	dict with the following keys:
						Sfix	-	list of lists, each containing [starttime]
						Ssac	-	list of lists, each containing [starttime]
						Sblk	-	list of lists, each containing [starttime]
						Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
						Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
						Eblk	-	list of lists, each containing [starttime, endtime, duration]
						msg	-	list of lists, each containing [time, message]
						NOTE: timing is in EDF time!
	
	arguments
	filename		-	path to the file that has to be read
	start		-	trial start string
	
	keyword arguments
	stop			-	trial ending string (default = None)
	missing		-	value to be used for missing data (default = 0.0)
	debug		-	Boolean indicating if DEBUG mode should be on or off;
				if DEBUG mode is on, information on what the script
				currently is doing will be printed to the console
				(default = False)
	
	returns
	data			-	a list with a dict for every trial (see above)
	"""

	# # # # #
	# debug mode
	
	def message(msg):
		if debug:
			print(msg)
	
	# # # # #
	# file handling
	
	# check if the file exists
	if not os.path.isfile(filename):
		raise Exception(f"Error in read_edf: file '{filename}' does not exist")
	
	# read file contents
	message(f"reading file '{filename}'")
	raw = open(filename, 'r').readlines()
	
	# # # # #
	# parse lines
	
	# Pre-compile frequently used string operations
	MSG_START = "MSG"
	SFIX_START = "SFIX"
	EFIX_START = "EFIX"
	SSACC_START = "SSACC"
	ESACC_START = "ESACC"
	SBLINK_START = "SBLINK"
	EBLINK_START = "EBLINK"

	# metadata
	metadata = {
		'CALIBRATION_TYPE': [], 'CALIBRATION_EYE': [], 'CALIBRATION_RESULT': [],
		'VALIDATION_TYPE': [], 'VALIDATION_EYE': [], 'VALIDATION_RESULT': [],
		'TRACKING_MODE': [], 'SAMPLING_RATE': [], 'FILE_SAMPLE_FILTER': [],
		'LINK_SAMPLE_FILTER': [], 'EYE_RECORDED': [], 'MOUNT_CONFIG': [],
		'GAZE_COORDS': [], 'PUPIL': [], 'PUPIL_TRACKING_ALGORITHM': []
	}
	
	# data
	data = []
	x, y, size, time, trackertime = [], [], [], [], []
	last_msg, last_msg_time = [], []
	events = {'Sfix': [], 'Ssac': [], 'Sblk': [], 'Efix': [], 'Esac': [], 'Eblk': [], 'msg': []}
	starttime = 0
	started = False
	trialend = False
	finalline = raw[-1]
	
	# loop through all lines
	for line in tqdm(raw, desc=f'Reading ASC file {filename}'):
		
		# store metadata
		if '!CAL CALIBRATION' in line:
			calibration_split = line.split()
			if calibration_split[-1] == 'ABORTED':
				metadata['CALIBRATION_TYPE'].append('ABORTED')
				metadata['CALIBRATION_EYE'].append(calibration_split[4])
				metadata['CALIBRATION_RESULT'].append('ABORTED')
			else:
				metadata['CALIBRATION_TYPE'].append(calibration_split[4])
				metadata['CALIBRATION_EYE'].append(calibration_split[5])
				metadata['CALIBRATION_RESULT'].append(calibration_split[7])
		elif '!CAL VALIDATION' in line:
			validation_split = line.split()
			if validation_split[-1] == 'ABORTED':
				metadata['VALIDATION_TYPE'].append('ABORTED')
				metadata['VALIDATION_EYE'].append(validation_split[4])
				metadata['VALIDATION_RESULT'].append('ABORTED')
			else:
				metadata['VALIDATION_TYPE'].append(validation_split[4])
				metadata['VALIDATION_EYE'].append(validation_split[5])
				metadata['VALIDATION_RESULT'].append(validation_split[7])
		elif 'RECCFG' in line:
			cfg_split = line.split()
			metadata['TRACKING_MODE'].append(cfg_split[3])
			metadata['SAMPLING_RATE'].append(cfg_split[4])
			metadata['FILE_SAMPLE_FILTER'].append(cfg_split[5])
			metadata['LINK_SAMPLE_FILTER'].append(cfg_split[6])
			metadata['EYE_RECORDED'].append(cfg_split[7])
		elif 'ELCLCFG' in line:
			elclcfg_split = line.split()
			metadata['MOUNT_CONFIG'].append(elclcfg_split[3])
		elif 'GAZE_COORDS' in line:
			gaze_coords_split = line.split()
			metadata['GAZE_COORDS'].append(gaze_coords_split[3:])
		elif 'ELCL_PROC' in line:
			elcl_proc_split = line.split()
			metadata['PUPIL_TRACKING_ALGORITHM'].append(elcl_proc_split[3])
		elif ('DIAMETER' in line) or ('AREA' in line):
			pupil_split = line.split()
			metadata['PUPIL'].append(pupil_split[1])
			
		# check if the current line contains start message
		if start in line:
			message(f"trialstart {len(data)}")
			started = True
			starttime = int(line.split()[1])
		
		if started:
			# message lines will start with MSG, followed by a tab, then a
			# timestamp, a space, and finally the message, e.g.:
			#	"MSG\t12345 something of importance here"
			if line.startswith(MSG_START):
				ms = line.find(" ") # message start
				t = int(line[4:ms]) # time
				m = line[ms+1:].strip() # message
				events['msg'].append([t, m]) 
		
			# EDF event lines are constructed of 9 characters, followed by
			# tab separated values; these values MAY CONTAIN SPACES, but
			# these spaces are ignored by float() (thank you Python!)
					
			# fixation start
			elif line.startswith(SFIX_START):
				message("fixation start")
				eye = line[5] # detect which eye
				l = line[9:]
				events['Sfix'].append([eye, int(l), m, t]) # also append the last message and its time
			# fixation end
			elif line.startswith(EFIX_START):
				message("fixation end")
				eye = line[5] # detect which eye
				l = line[9:].split('\t')
				st, et, dur = map(int, l[:3])
				sx, sy = map(lambda v: replace_missing(v, missing=missing), l[3:5])
				events['Efix'].append([eye, st, et, dur, sx, sy, m, t]) # also append the last message and its time
			# saccade start
			elif line.startswith(SSACC_START):
				message("saccade start")
				eye = line[6] # detect which eye
				l = line[9:]
				events['Ssac'].append([eye, int(l), m, t]) # also append the last message and its time
			# saccade end
			elif line.startswith(ESACC_START):
				message("saccade end")
				eye = line[6] # detect which eye
				l = line[9:].split('\t')
				st, et, dur = map(int, l[:3])
				sx, sy, ex, ey = map(lambda v: replace_missing(v, missing=missing), l[3:7])
				ampl, pv = l[7], l[8]
				events['Esac'].append([eye, st, et, dur, sx, sy, ex, ey, ampl, pv, m, t]) # also append the last message and its time
			# blink start
			elif line.startswith(SBLINK_START):
				message("blink start")
				eye = line[7] # detect which eye
				l = line[9:]
				events['Sblk'].append([eye, int(l), m, t]) # also append the last message and its time
			# blink end
			elif line.startswith(EBLINK_START):
				message("blink end")
				eye = line[7] # detect which eye
				l = line[9:].split('\t')
				st, et, dur = map(int, l[:3])
				events['Eblk'].append([eye, st, et, dur, m, t]) # also append the last message and its time


			# regular lines will contain tab separated values, beginning with
			# a timestamp, followed by the values that were asked to be stored
			# in the EDF and a mysterious '...'. Usually, this comes down to
			# timestamp, x, y, pupilsize, ...
			# e.g.: "985288\t  504.6\t  368.2\t 4933.0\t..."
			# NOTE: these values MAY CONTAIN SPACES, but these spaces are
			# ignored by float() (thank you Python!)
			else:
				# see if current line contains relevant data
				try:
					# split by tab
					l = line.split('\t')
					# if first entry is a timestamp, this should work
					int(l[0])
				except ValueError:
					message(f"line '{line}' could not be parsed")
					continue # skip this line

				# check missing
				if float(l[3]) == 0.0:
					l[1] = 0.0
					l[2] = 0.0
				
				# extract data
				x.append(float(l[1]))
				y.append(float(l[2]))
				size.append(float(l[3]))
				time.append(int(l[0]) - starttime)
				trackertime.append(int(l[0]))
				last_msg.append(m)
				last_msg_time.append(t)

		# Moved checking trial end to the end of the loop to allow the last line (stop MSG) to be extracted
		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop is not None:
				if stop in line:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if (start in line) or (line == finalline):
					started = True
					trialend = True
			
			# # # # #
			# trial ending
			
			if trialend:
				message(f"trialend {len(data)}; {len(x)} samples found")
				# trial dict
				trial = {
					'x': numpy.array(x, dtype=float),
					'y': numpy.array(y, dtype=float),
					'size': numpy.array(size, dtype=float),
					'time': numpy.array(time, dtype=int),
					'trackertime': numpy.array(trackertime, dtype=int),
					'last_msg': last_msg[:],
					'last_msg_time': last_msg_time[:],
					'events': {k: v[:] for k, v in events.items()}
				}
				# add trial to data
				data.append(trial)
				# reset stuff
				x.clear()
				y.clear()
				size.clear()
				time.clear()
				trackertime.clear()
				last_msg.clear()
				last_msg_time.clear()
				for k in events:
					events[k].clear()
				trialend = False
			
	# # # # #
	# return
	
	return data, metadata