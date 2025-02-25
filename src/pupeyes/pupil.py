# -*- coding:utf-8 -*-

"""
Eyelink Pupil Data Processing Module

Author: Han Zhang
Email: hanzh@umich.edu

This module provides tools for processing pupillometry data from Eyelink eye trackers.
It includes functionality for deblinking, smoothing, baseline correction, and plotting
pupil size data.
"""

import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from .utils import make_mask, lowpass_filter
from .aoi import is_inside
from .external.based_noise_blinks_detection import based_noise_blinks_detection

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from .defaults import default_mpl, default_plotly

class PupilProcessor:

    def __init__(self, data, trial_identifier, pupil_col, time_col, x_col, y_col, samp_freq, convert_pupil_size=True, artificial_d=5, artificial_size=5663, recording_unit='diameter'):
        """
        Initialize PupilData object.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing pupil size data
        trial_identifier : str or list
            Column name(s) for trial identifier
        pupil_col : str
            Column name for pupil size
        time_col : str
            Column name for time
        x_col : str
            Column name for x gaze position
        y_col : str
            Column name for y gaze position
        samp_freq : int
            Sampling frequency in Hz
        """
        # make a copy of the data
        self.data = data.copy() 
        # group by column for preprocessing
        if isinstance(trial_identifier, str):
            self.trial_identifier = [trial_identifier]
        else:
            self.trial_identifier = trial_identifier
        # column name for pupil size
        self.pupil_col = pupil_col 
        # column name for time
        self.time_col = time_col 
        # column name for x gaze position
        self.x_col = x_col 
        # column name for y gaze position
        self.y_col = y_col 
        # sampling frequency
        self.samp_freq = samp_freq
        # store all preprocessing steps
        self.all_steps = [] 
        # store generated pupil columns
        self.all_pupil_cols = [pupil_col]
        # store parameters for each step
        self.params = dict()
        # trials
        self.trials = self.data[self.trial_identifier].drop_duplicates().reset_index(drop=True)
        # empty dataframe to store summary of preprocessing steps
        self.summary_data = self.data.groupby(self.trial_identifier, sort=False).size().reset_index(name='n_samples')
        # outlier detection by info. leave as None if not performed
        self.baseline_outlier_by = None
        self.trace_outlier_by = None

        # check if the difference between consecutive samples is equal to a fixed value
        diff = self.data.groupby(self.trial_identifier, sort=False)[self.time_col].diff().dropna().unique()
        if len(diff) == 1:
           if 1000/diff[0] != self.samp_freq:
                raise ValueError(f'Actual sampling frequency {1000/diff[0]}Hz does not match the provided sampling frequency {self.samp_freq}Hz')
        else:
            raise ValueError('Sampling frequency is not consistent')

        # convert pupil size
        if convert_pupil_size:
            self.data[self.pupil_col] = convert_pupil(self.data[self.pupil_col], artificial_d=artificial_d, artificial_size=artificial_size, recording_unit=recording_unit)
            print(f'Pupil data converted to {recording_unit} with artificial d={artificial_d} and artificial size={artificial_size}')

        print(f'PupilProcessor initialized with {len(self.data)} samples')
        print(f'Pupil column: {self.pupil_col}, Time column: {self.time_col}, X column: {self.x_col}, Y column: {self.y_col}')
        print(f'Trial identifier: {self.trial_identifier}, Number of trials: {len(self.trials)}')


    def deblink(self, suffix='_db'):
        """
        Remove blinks from pupil data using noise-based blink detection.

        Uses the based_noise_blinks_detection algorithm to identify and remove blinks from the pupil data.
        Creates a new column with the deblinked data.

        Parameters
        ----------
        suffix : str, default='_db'
            Suffix to append to the pupil column name for the deblinked data

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with deblink statistics
        - Adds new column with suffix to data
        - Updates all_pupil_cols and all_steps
        """
        # store parameters
        self.params['deblink'] = {k:v for k,v in locals().items() if k != 'self'}

        # create new column for deblinked data
        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix
        self.data[new_col] = self.data[pupil_col] # default to last pupil column

        # get sampling frequency
        samp_freq = self.samp_freq

        # initialize blinks removed column in summary data
        self.summary_data['run_deblink'] = False
        self.summary_data['pct_deblink'] = pd.NA

        # iterate over trials if trial_identifier is provided
        empty_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Deblinking'):
            # check if groupdata has any pupil data
            if np.all(groupdata[pupil_col].isna()):
                empty_trials.append(group)
            else:
                # detect and remove blinks 
                blinks = based_noise_blinks_detection(groupdata[new_col].values, sampling_freq=samp_freq)
                for onset, offset in zip(blinks['blink_onset'], blinks['blink_offset']):
                    # select row numbers in the group data, which are used to set the same row numbers in the full data to NA
                    self.data.loc[groupdata.index[int(onset):int(offset)], new_col] = pd.NA

                # update summary data
                nblink = len(blinks["blink_onset"])
                nblinksamps = int(np.sum(np.array(blinks["blink_offset"]) - np.array(blinks["blink_onset"])))
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_deblink'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_deblink'] = nblinksamps/len(groupdata)

        # replace potential other 0 values with NaN
        self.data[new_col] = self.data[new_col].replace({0:pd.NA})

        # update latest pupil column 
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Deblinked')

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {len(empty_trials)} trials not deblinked due to missing pupil data:")
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self

    def artifact_rejection(self, suffix='_ar', method='both', speed_n=16, zscore_threshold=2.5, zscore_allowp=0.1):
        """
        Reject artifacts from pupil data using speed and/or z-score based methods.

        Identifies and removes artifacts using pupil speed and/or z-score thresholds.
        Creates a new column with the artifact-rejected data.

        Parameters
        ----------
        suffix : str, default='_ar'
            Suffix to append to the pupil column name for the artifact-rejected data
        method : {'speed', 'zscore', 'both'}, default='both'
            Method to use for artifact rejection
        speed_n : int, default=16
            Number of MADs above median speed to use as threshold
        zscore_threshold : float, default=2.5
            Z-score threshold for artifact rejection
        zscore_allowp : float, default=0.1
            Proportion of mean to use as minimum standard deviation

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with artifact rejection statistics
        - Adds new column with suffix to data
        - Updates all_pupil_cols and all_steps
        """
        # store parameters
        self.params['artifact_rejection'] = {k:v for k,v in locals().items() if k != 'self'}

        # create new column for artifact rejected data
        time_col = self.time_col
        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix
        self.data[new_col] = self.data[pupil_col] # default to last pupil column
        
        # initialize artifacts removed column in summary data
        if method in ['speed', 'both']:
            self.summary_data['run_speed'] = False
            self.summary_data['pct_speed'] = pd.NA
        if method in ['zscore', 'both']:
            self.summary_data['run_size'] = False
            self.summary_data['pct_size'] = pd.NA

        # iterate over trials if trial_identifier is provided
        empty_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Artifact rejection'):

            # check if groupdata has any pupil data
            if np.all(groupdata[pupil_col].isna()):
                empty_trials.append(group)
            else:
                if method in ['speed', 'both']:
                    speed_mask = np.zeros(len(groupdata), dtype=bool) # initialize mask
                    pupil_speed = compute_speed(groupdata[pupil_col].values, groupdata[time_col].values)
                    median_speed = np.nanmedian(pupil_speed)
                    mad = np.nanmedian(np.abs(pupil_speed - median_speed))
                    speed_mask = pupil_speed > (median_speed + (speed_n * mad))
                    # select row numbers in the group data, which are then used in .loc to select values needed to be set to nan
                    self.data.loc[groupdata.index[speed_mask], new_col] = pd.NA
                    # calculate percentage of speed artifacts
                    pct_speed_artifacts = speed_mask.sum()/len(speed_mask)
                    # update summary data
                    self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_speed'] = True
                    self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_speed'] = pct_speed_artifacts
                    
                if method in ['zscore', 'both']:
                    zscore_mask = np.zeros(len(groupdata), dtype=bool) # initialize mask
                    mean = np.nanmean(groupdata[new_col])
                    std = np.nanstd(groupdata[new_col])
                    # check if std is larger than zscore_allowp * mean
                    if std > zscore_allowp * mean:
                        zscore_mask = np.abs(groupdata[new_col] - mean) > zscore_threshold * std
                        self.data.loc[groupdata.index[zscore_mask], new_col] = pd.NA
                        # calculate percentage of size artifacts
                        pct_size_artifacts = zscore_mask.sum()/len(zscore_mask)
                    else:
                        pct_size_artifacts = 0.0 # no size artifacts

                    # update summary data
                    self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_size'] = True
                    self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_size'] = pct_size_artifacts

        # update latest pupil column
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Artifact Rejected')

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {len(empty_trials)} trials not artifact rejected due to missing pupil data:")
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self

    def filter_position(self, suffix = '_xy', vertices=[(0,0), (0, 1080), (1920, 1080), (1920, 0), (0,0)]):
        """
        Filter pupil data based on gaze position within a polygon.

        Removes pupil data points where gaze position falls outside a specified polygon.
        Creates a new column with the position-filtered data.

        Parameters
        ----------
        suffix : str, default='_xy'
            Suffix to append to the pupil column name for the filtered data
        vertices : list of tuples, default=[(0,0), (0,1080), (1920,1080), (1920,0), (0,0)]
            List of (x,y) coordinates defining the polygon vertices

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with position filtering statistics
        - Adds new column with suffix to data
        - Updates all_pupil_cols and all_steps
        """
        # check if vertices can be converted to float numpy array
        try:
            vertices = np.array(vertices, dtype=float)
        except:
            raise ValueError("Vertices must be convertible to float numpy array")

        # store parameters
        self.params['filter_position'] = {k:v for k,v in locals().items() if k != 'self'}
        
        # create new column for filtered gaze position data
        x_col = self.x_col
        y_col = self.y_col
        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix
        self.data[new_col] = self.data[pupil_col]

        # initialize position removed column in summary data
        self.summary_data['run_position'] = False
        self.summary_data['pct_position'] = pd.NA

        # iterate over trials if trial_identifier is provided
        empty_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Filtering based on gaze position'):
            # check if groupdata has any pupil data
            if np.all(groupdata[pupil_col].isna()):
                empty_trials.append(group)
            else:
                # get gaze position
                gaze_pos = np.array(groupdata[[x_col, y_col]], dtype=float)
                # check if gaze position is inside the specified region
                inside_mask = is_inside(gaze_pos, vertices)
                # set pupil size to NaN if gaze position is outside the specified region
                self.data.loc[groupdata.index[~inside_mask], new_col] = pd.NA

                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_position'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_position'] = 1 - (inside_mask.sum()/len(inside_mask))
                
        # update latest pupil column
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Position Filtered')

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {len(empty_trials)} trials not filtered due to missing pupil data:")
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self

    def smooth(self, suffix='_sm', method='hann', window=100, **kwargs):
        """
        Smooth pupil data using various smoothing methods.

        Applies smoothing to pupil data using rolling mean, Hann window, or Butterworth filter.
        Creates a new column with the smoothed data.

        Parameters
        ----------
        suffix : str, default='_sm'
            Suffix to append to the pupil column name for the smoothed data
        method : {'rollingmean', 'hann', 'butter'}, default='hann'
            Method to use for smoothing
        window : int, default=100
            Window size for rolling mean or Hann window smoothing
        **kwargs : dict
            Additional arguments passed to the smoothing functions

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - For Butterworth filter, sampling_freq and cutoff_freq must be specified in kwargs
        - Updates summary_data with smoothing statistics
        - Adds new column with suffix to data
        - Updates all_pupil_cols and all_steps
        """
        # store parameters
        self.params['smooth'] = {k:v for k,v in locals().items() if k != 'self'}

        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix

        if not isinstance(window, int) or window < 3:
            raise ValueError("Window size must be integer >= 3")
            
        if method not in ['rollingmean', 'hann', 'butter']:
            raise ValueError("Method must be 'rollingmean', 'hann', or 'butter'")

        if (method in ['rollingmean', 'hann']) and (len(self.data[pupil_col]) < window):
            raise ValueError('Data length smaller than window size')

        if method == 'butter' and ('sampling_freq' not in kwargs or 'cutoff_freq' not in kwargs):
            raise ValueError("For Butterworth filter, 'sampling_freq' and 'cutoff_freq' must be specified")
        
        if (method == 'butter') and (self.data[pupil_col].isnull().sum() > 0):
            raise ValueError("Butterworth filter does not support NaN values")
        
        # create new column for smoothed data
        self.data[new_col] = pd.NA

        # initialize summary data
        self.summary_data['run_smooth'] = False

        # iterate over trials if trial_identifier is provided
        empty_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Smoothing'):

            # check if groupdata has any pupil data
            if np.all(groupdata[pupil_col].isna()):
                empty_trials.append(group)
            else:
                if method == 'rollingmean':
                    smoothed = groupdata[pupil_col].rolling(window=window, center=True, **kwargs).mean()
                elif method == 'hann':
                    smoothed = groupdata[pupil_col].rolling(window=window, win_type='hann', center=True, **kwargs).mean()
                elif method == 'butter':
                    smoothed = lowpass_filter(groupdata[pupil_col], **kwargs)
                self.data.loc[groupdata.index, new_col] = smoothed

                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_smooth'] = True

        # convert to Float64 since some values changed to float64
        self.data[new_col] = self.data[new_col].convert_dtypes()

        # update latest pupil column
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Smoothed')

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {len(empty_trials)} trials not smoothed due to missing pupil data:")
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self


    def check_missing(self, pupil_col=None, missing_value=pd.NA):
        """
        Check for missing values in pupil data.

        Calculates the percentage of missing values for each trial.
        Updates summary data with missing value statistics.

        Parameters
        ----------
        pupil_col : str, optional
            Column name to check for missing values. If None, uses latest pupil column
        missing_value : float or pd.NA, default=pd.NA
            Value to consider as missing

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with missing value statistics
        - Updates all_steps
        """
        # store parameters
        self.params['check_missing'] = {k:v for k,v in locals().items() if k != 'self'}

        # use latest pupil column if not specified
        if pupil_col is None:
            pupil_col = self.all_pupil_cols[-1] 

        # initialize summary data
        self.summary_data['run_check_missing'] = False
        self.summary_data['missing'] = 0.0

        # iterate over trials if trial_identifier is provided
        skip_trials = []
        missing_pct = 0.0
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Checking missing values'):
            try:
                if pd.isna(missing_value):
                    missing_pct = groupdata[pupil_col].isna().sum()/len(groupdata)
                else:
                    missing_pct = (groupdata[pupil_col] == missing_value).sum()/len(groupdata)
                
                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_check_missing'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'missing'] = missing_pct
            except:
                skip_trials.append(group)
            
        # update latest step
        self.all_steps.append('Missing Values Checked')
        
        # print failed trials
        if len(skip_trials) > 0:
            print(f"\n {len(skip_trials)} trials not checked due to missing data:")
            print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

        return self


    def interpolate(self, suffix='_it', method='linear', missing_threshold=0.6):
        """
        Interpolate missing values in pupil data.

        Fills missing values using linear or spline interpolation.
        Creates a new column with the interpolated data.

        Parameters
        ----------
        suffix : str, default='_it'
            Suffix to append to the pupil column name for the interpolated data
        method : {'linear', 'spline'}, default='linear'
            Method to use for interpolation
        missing_threshold : float, default=0.6
            Maximum proportion of missing values allowed for interpolation

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with interpolation statistics
        - Adds new column with suffix to data
        - Updates all_pupil_cols and all_steps
        """
        if method not in ['spline', 'linear']:
            raise ValueError("Invalid method. Use 'linear' or 'spline'")

        # store parameters
        self.params['interpolate'] = {k:v for k,v in locals().items() if k != 'self'}

        # create new column for interpolated data
        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix
        
        # initialize summary data
        self.summary_data['run_interpolate'] = False
        self.summary_data['pct_interpolate'] = 0.0

        # iterate over trials if trial_identifier is provided
        skip_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Interpolating'):

            # update summary data
            pct_missing = groupdata[pupil_col].isna().mean()
            self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_interpolate'] = pct_missing
            
            # check for missing values
            if (pct_missing >= missing_threshold): 
                skip_trials.append(group)
            else:
                if method == 'linear':
                    interpolated = groupdata[pupil_col].interpolate(method='linear').ffill().bfill()
                else:
                    interpolated = groupdata[pupil_col].interpolate(method='spline', order=3).ffill().bfill()
                # overwrite the new column with interpolated values
                self.data.loc[groupdata.index, new_col] = interpolated
                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_interpolate'] = True
            
        # update latest pupil column
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Interpolated')

        if len(skip_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {len(skip_trials)} trials not interpolated due to high missing values:")
            print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

        return self

    def resample(self, bin_size_ms=10, agg_methods=None):
        """
        Resample pupil data to a new sampling rate.

        Resamples data by binning into fixed time windows and aggregating values.
        Updates the data with resampled values.

        Parameters
        ----------
        bin_size_ms : int, default=10
            Size of time bins in milliseconds
        agg_methods : dict, optional
            Dictionary mapping column names to aggregation methods

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates data with resampled values
        - Updates summary_data with resampling statistics
        - Updates all_steps
        """
        # store parameters
        self.params['resample'] = {k: v for k, v in locals().items() if k != 'self'}

        # get data
        data = self.data

        # create new column for resampled data
        time_col = self.time_col

        # aggregate methods for resampling
        aggregation_methods = {col: 'first' for col in data.columns}
        if agg_methods is not None:
            aggregation_methods.update(agg_methods)

        # initialize summary data
        self.summary_data['run_resample'] = False

        # group data once
        grouped = self.data.groupby(self.trial_identifier, sort=False)

        # precompute offsets for each group
        offsets = grouped[time_col].transform('min')

        # normalize time and compute bins outside the loop
        normalized_time = self.data[time_col] - offsets
        bins = normalized_time // bin_size_ms

        # iterate over trials and aggregate data
        skip_trials = []
        all_resampled = []
        for group, groupdata in tqdm(grouped, desc='Resampling'):
            try:
                # group by bins and aggregate data
                groupdata = groupdata.groupby(bins, as_index=False).agg(aggregation_methods)
                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_resample'] = True
            except:
                skip_trials.append(group)

            # append resampled data
            all_resampled.append(groupdata)

        # concatenate resampled data
        data = pd.concat(all_resampled, ignore_index=True)

        # update data
        self.data = data

        # print failed trials
        if len(skip_trials) > 0:
            print(f"\n Failed to resample {len(skip_trials)} trials:")
            print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

        # update latest step
        self.all_steps.append('Resampled')

        return self
        

    def baseline_correction(self, baseline_query, baseline_range=[None, None], suffix='_bc', method='subtractive'):
        """
        Apply baseline correction to pupil data.

        Corrects pupil data by subtracting or dividing by baseline values.
        Creates a new column with the baseline-corrected data.

        Parameters
        ----------
        baseline_query : str
            Query string to select baseline period data
        baseline_range : list, default=[None, None]
            Start and end indices for baseline period
        suffix : str, default='_bc'
            Suffix to append to the pupil column name for the corrected data
        method : {'subtractive', 'divisive'}, default='subtractive'
            Method to use for baseline correction

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with baseline correction statistics
        - Adds new column with suffix to data
        - Updates all_pupil_cols and all_steps
        """
        # check for valid method
        if method not in ['subtractive', 'divisive']:
            raise ValueError("Invalid method. Use 'subtractive' or 'divisive'")

        # store parameters
        self.params['baseline_correction'] = {k:v for k,v in locals().items() if k != 'self'}

        # initialize summary data
        self.summary_data['run_baseline_correction'] = False
        self.summary_data['baseline'] = pd.NA

        # which columns to use for baseline correction
        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix

        # get baseline data
        baseline_data = self.data.query(baseline_query)

        # get baseline range    
        s, e = baseline_range

        # Precompute baseline means for each group
        baseline_means = baseline_data.groupby(self.trial_identifier)[pupil_col].apply(lambda x: x.iloc[s:e].mean())

        # iterate over trials in data
        skip_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Baseline correction'):
            # select baseline data for the current group
            baseline = baseline_means.loc[group]

            # check for nan
            if pd.isna(baseline) or np.all(groupdata[pupil_col].isna()):
                skip_trials.append(group)
            else:
                # do baseline correction
                if method == 'subtractive':
                    self.data.loc[groupdata.index, new_col] = groupdata[pupil_col] - baseline
                else:
                    self.data.loc[groupdata.index, new_col] = groupdata[pupil_col] / baseline

                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_baseline_correction'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'baseline'] = baseline

        # print trials not baseline corrected
        if len(skip_trials) > 0:
            print(f"\n {len(skip_trials)} trials not baseline corrected due to missing data:")
            print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

        # update latest step and latest pupil column
        self.all_steps.append('Baseline Corrected')
        self.all_pupil_cols.append(new_col)

        return self

    def check_baseline_outliers(self, outlier_by=None, n_mad_baseline=4, plot=True, **kwargs):
        """
        Check for outliers in baseline pupil values.

        Identifies outliers in baseline values using median absolute deviation (MAD).
        Can group data and check outliers within groups.

        Parameters
        ----------
        outlier_by : str or list, optional
            Column(s) to group data by for outlier detection
        n_mad_baseline : float, default=4
            Number of MADs from median to use as outlier threshold
        plot : bool, default=True
            Whether to plot the baseline distributions
        **kwargs : dict
            Additional arguments passed to plot_baseline

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with baseline outlier statistics
        - Updates all_steps
        """
        # get summary data
        df_summary = self.summary_data.copy()

        # check if baseline data is available
        if 'baseline' not in df_summary.columns:
            raise ValueError("Baseline data is not available. Please run baseline correction first.")
        
        # convert outlier_by to list if it is a string
        if isinstance(outlier_by, str):
            outlier_by = [outlier_by]

        # initialize outlier masks
        df_summary['baseline_outlier'] = False
        df_summary['baseline_upper'] = pd.NA
        df_summary['baseline_lower'] = pd.NA

        if outlier_by is None:
            # calculate thresholds using pandas
            median_baseline = df_summary['baseline'].median()
            mad = (df_summary['baseline'] - median_baseline).abs().median()
            
            # calculate thresholds
            upper = median_baseline + n_mad_baseline*mad
            lower = median_baseline - n_mad_baseline*mad
            
            # mark outliers
            df_summary['baseline_outlier'] = (df_summary['baseline'] > upper) | (df_summary['baseline'] < lower)
            df_summary['baseline_upper'] = upper
            df_summary['baseline_lower'] = lower
        else:
            # calculate thresholds for each group
            for group, groupdata in tqdm(df_summary.groupby(outlier_by, sort=False), desc='Checking baseline pupil sizes for outliers'):
                # calculate group thresholds using pandas
                median_baseline = groupdata['baseline'].median()
                mad = (groupdata['baseline'] - median_baseline).abs().median()
                upper = median_baseline + n_mad_baseline*mad
                lower = median_baseline - n_mad_baseline*mad
                
                # mark outliers for this group
                group_indices = groupdata.index
                df_summary.loc[group_indices, 'baseline_outlier'] = (groupdata['baseline'] > upper) | (groupdata['baseline'] < lower)
                df_summary.loc[group_indices, 'baseline_upper'] = upper
                df_summary.loc[group_indices, 'baseline_lower'] = lower

        # print summary
        print(f"\n {df_summary['baseline_outlier'].sum()} trials detected as baseline outliers:")
        if df_summary['baseline_outlier'].any():
            print(f"\n {df_summary.query('baseline_outlier==True')[['subject','block','trial']]}")

        # update summary data
        self.summary_data = df_summary

        # update outlier by
        self.baseline_outlier_by = outlier_by

        # update steps
        self.all_steps.append('Baseline Outlier Detection')
        
        # plot if requested
        if plot:
            self.plot_baseline(plot_by=outlier_by, return_fig=False, **kwargs)

        return self
        
    def check_trace_outliers(self, x=None, y=None, outlier_by=None, n_mad_trace=4, plot=True, **kwargs):
        """
        Check for outlier trials based on pupil trace values.

        Detects outlier trials by comparing each trial's pupil trace against thresholds calculated from the median absolute deviation (MAD) of all trials.
        Outliers can be calculated globally or within specified groups.

        Parameters
        ----------
        x : str, optional
            Column name for x-axis values (time). Defaults to time column.
        y : str, optional 
            Column name for pupil values. Defaults to last pupil column.
        outlier_by : str or list, optional
            Column(s) to group trials by when calculating outlier thresholds.
        n_mad_trace : float, default=4
            Number of MADs to use for outlier threshold.
        plot : bool, default=True
            Whether to plot the results.
        **kwargs
            Additional arguments passed to plotting function.

        Returns
        -------
        self : object
            Returns self for method chaining.

        Notes
        -----
        - Adds 'trace_outlier', 'trace_upper', 'trace_lower' columns to summary data
        - Outlier detection uses median absolute deviation (MAD) method
        - Can detect outliers globally or within groups specified by outlier_by
        """

        # get data
        df = self.data.copy()
        df_summary = self.summary_data.copy()

        # get x and y columns
        if x is None:
            x = self.time_col
        if y is None:
            y = self.all_pupil_cols[-1]
        print(f'Checking trace outliers for {y}')

        # initialize outlier columns
        df_summary['trace_outlier'] = False
        df_summary['trace_upper'] = pd.NA
        df_summary['trace_lower'] = pd.NA

        # calculate outlier thresholds
        if outlier_by is None:
            # calculate thresholds for all trials
            grand_mean = df[y].mean()
            pupil_dist = df.groupby(self.trial_identifier)[y].apply(lambda x: (x - grand_mean).abs().max())
            median_dist = pupil_dist.median()
            mad = (pupil_dist - median_dist).abs().median()
            
            upper = grand_mean + median_dist + (n_mad_trace * mad)
            lower = grand_mean - median_dist - (n_mad_trace * mad)

            # update summary data
            df_summary['trace_upper'] = upper
            df_summary['trace_lower'] = lower
        else:
            if not isinstance(outlier_by, list):
                outlier_by = [outlier_by]
            
            # calculate thresholds for each group
            for group, groupdata in df.groupby(outlier_by, sort=False):
                grand_mean = groupdata[y].mean()
                pupil_dist = groupdata.groupby(self.trial_identifier)[y].apply(lambda x: (x - grand_mean).abs().max())
                median_dist = pupil_dist.median()
                mad = (pupil_dist - median_dist).abs().median()
                
                upper = grand_mean + median_dist + (n_mad_trace * mad)
                lower = grand_mean - median_dist - (n_mad_trace * mad)

                # update summary data for this group
                df_summary.loc[np.all(df_summary[outlier_by] == group, axis=1), 'trace_upper'] = upper
                df_summary.loc[np.all(df_summary[outlier_by] == group, axis=1), 'trace_lower'] = lower

        # mark outliers
        for trial, trialdata in tqdm(df.groupby(self.trial_identifier, sort=False), desc='Checking pupil traces for outliers'):
            # get max and min values
            max_val = trialdata[y].max()
            min_val = trialdata[y].min()

            # get thresholds for this trial
            trial_mask = np.all(df_summary[self.trial_identifier] == trial, axis=1)
            upper_threshold = df_summary.loc[trial_mask, 'trace_upper'].iloc[0]
            lower_threshold = df_summary.loc[trial_mask, 'trace_lower'].iloc[0]

            # check for outliers and update summary data
            if pd.notna(max_val) and pd.notna(min_val) and pd.notna(upper_threshold) and pd.notna(lower_threshold):
                is_outlier = (max_val > upper_threshold) or (min_val < lower_threshold) 
                df_summary.loc[trial_mask, 'trace_outlier'] = is_outlier

        # print outlier trials
        outlier_trials = df_summary.query('trace_outlier==True')[self.trial_identifier]
        if len(outlier_trials) > 0:
            print(f"\n {len(outlier_trials)} trials detected as outliers:")
            print(f"\n {pd.DataFrame(outlier_trials)}")

        # update summary data and steps
        self.summary_data = df_summary
        self.all_steps.append('Trace Outlier Detection')

        # update outlier by
        self.trace_outlier_by = outlier_by

        # plot if requested
        if plot:
            self.plot_spaghetti(x=x, y=y, plot_by=outlier_by, return_fig=False, **kwargs)

        return self

    def summary(self, columns=None, level=None, agg_methods=None):
        """
        Get summary statistics of the data.

        Returns summary data for specified columns, optionally grouped by level and aggregated using specified methods.

        Parameters
        ----------
        columns : list, optional
            Columns to include in summary. Defaults to all columns.
        level : str or list, optional
            Column(s) to group by.
        agg_methods : dict, optional
            Dictionary mapping column names to aggregation methods.
            If None, uses mean for numeric columns.

        Returns
        -------
        pandas.DataFrame
            Summary statistics dataframe.
        """
        if columns is None:
            columns = self.summary_data.columns
        if level is None:
            return self.summary_data[columns]
        else:
            if agg_methods is None:
                # get all numeric columns
                numeric_cols = self.summary_data.select_dtypes(include=['number','boolean']).columns
                agg_methods = {col: 'mean' for col in numeric_cols}
                print(f"Using default aggregation methods: {agg_methods}")
            return self.summary_data.groupby(level).agg(agg_methods)


    def validate_trials(self, trials_to_exclude, invert_mask=False):
        """
        Mark trials as valid/invalid based on exclusion criteria.

        Parameters
        ----------
        trials_to_exclude : pandas.DataFrame
            DataFrame containing trial identifiers to exclude.
        invert_mask : bool, default=False
            If True, excludes all trials except those specified.

        Returns
        -------
        self : object
            Returns self for method chaining.

        Notes
        -----
        Adds 'valid' column to both summary_data and data.
        """

        # drop duplicates
        trials_to_exclude = trials_to_exclude.drop_duplicates()

        # get mask
        summary_mask = make_mask(self.summary_data, trials_to_exclude, invert=invert_mask)
        data_mask = make_mask(self.data, trials_to_exclude, invert=invert_mask)

        # update summary data
        self.summary_data['valid'] = summary_mask
        # update data
        self.data['valid'] = data_mask

        return self

    def _get_plot_settings(self, x, y, plot_params=None, is_interactive=True):
        """
        Helper method to get plot settings for both static and interactive trial plots.

        Parameters
        ----------
        x : str
            Column name for x-axis values
        y : list
            Column name(s) for y-axis values
        plot_params : dict, optional
            Dictionary of plotting parameters to override defaults
        is_interactive : bool, default=True
            Whether settings are for interactive (Plotly) or static (Matplotlib) plot

        Returns
        -------
        tuple
            (plot_specific_settings, kwargs) where kwargs are either matplotlib or plotly settings
        """
        if plot_params is None:
            plot_params = {}

        # common plot specific settings
        plot_specific_settings = {
            'layout': (len(y), 1),  # number of rows, number of columns
            'subplot_titles': y,  # subplot titles
            'x_title': x,
            'y_title': '',
            'showlegend': True,
            'grid': False,  # show grid
        }
        
        # update plot-specific settings if provided
        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings})

        if is_interactive:
            # plotly specific settings
            kwargs = default_plotly.copy()
            kwargs['width'] = plot_params.get('width', 800*plot_specific_settings['layout'][1])
            kwargs['height'] = plot_params.get('height', 300*plot_specific_settings['layout'][0])
            kwargs['title_text'] = plot_params.get('title_text', '')
            kwargs['xaxis_title'] = plot_params.get('xaxis_title', plot_specific_settings['x_title'])
            kwargs['yaxis_title'] = plot_params.get('yaxis_title', plot_specific_settings['y_title'])
            kwargs['showlegend'] = plot_params.get('showlegend', plot_specific_settings['showlegend'])
            kwargs['xaxis_showgrid'] = plot_params.get('xaxis_showgrid', plot_specific_settings['grid'])
            kwargs['yaxis_showgrid'] = plot_params.get('yaxis_showgrid', plot_specific_settings['grid'])
        else:
            # matplotlib specific settings
            kwargs = default_mpl.copy()
            kwargs['figure.figsize'] = plot_params.get('figure.figsize', 
                                                     (10*plot_specific_settings['layout'][1], 
                                                      3*plot_specific_settings['layout'][0]))
        
        # update with any remaining valid kwargs
        kwargs.update({k:v for k,v in plot_params.items() if k in kwargs})
        
        return plot_specific_settings, kwargs

    def plot_trial(self, trial, x=None, y=None, hue=None, save=None, interactive=True, plot_params=None):
        """
        Plot data for a single trial.

        A wrapper function that calls either _plot_trial_interactive() or _plot_trial_static() 
        depending on the interactive parameter.

        Parameters
        ----------
        trial : pandas.DataFrame
            DataFrame containing trial identifier.
        x : str, optional
            Column name for x-axis values. Defaults to time column.
        y : str or list, optional
            Column name(s) for y-axis values. Defaults to all pupil columns.
        hue : str or list, optional
            Column(s) to group data by for separate lines.
        save : str, optional
            Path to save plot.
        interactive : bool, default=True
            Whether to create interactive plot.
        plot_params : dict, optional
            Additional plotting parameters.

        Returns
        -------
        figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Plot figure object.
        axes : matplotlib.axes.Axes, optional
            Plot axes object (only for static plots).
        """
        if plot_params is None:
            plot_params = {}

        # plot using appropriate function
        if interactive:
            return self._plot_trial_interactive(trial, x, y, hue, save, plot_params)
        else:
            return self._plot_trial_static(trial, x, y, hue, save, plot_params)

    def _plot_trial_static(self, trial, x=None, y=None, hue=None, save=None, plot_params=None):
        """
        Create static plot of trial data using matplotlib.

        Parameters
        ----------
        trial : pandas.DataFrame
            DataFrame containing trial identifier.
        x : str, optional
            Column name for x-axis values. Defaults to time column.
        y : str or list, optional
            Column name(s) for y-axis values. Defaults to all pupil columns.
        hue : str or list, optional
            Column(s) to group data by for separate lines
        save : str, optional
            Path to save figure
        plot_params : dict, optional
            Dictionary of plotting parameters to override defaults. Can include:
            - layout : tuple of (rows, cols) for subplot layout
            - subplot_titles : list of titles for subplots
            - x_title : x-axis label
            - y_title : y-axis label
            - showlegend : bool, whether to show legend
            - grid : bool, whether to show grid
            - Any matplotlib rcParams key

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects (fig, ax).
        
        Notes
        -----
        - Uses matplotlib for static plotting
        - Creates subplots if multiple y columns provided
        - Groups data by hue variable(s) if provided
        - Applies default matplotlib styling that can be overridden
        """
        if plot_params is None:
            plot_params = {}

        # get data
        data = self.data.copy()

        # get mask
        mask = make_mask(data, trial, invert=True)

        # mask data
        data = data[mask]

        # check if data is empty
        if data.empty:
            trial_info = ', '.join(f"{k}: {v}" for k, v in trial.items())
            raise ValueError(f"No data found for trial with {trial_info}")

        # get x and y
        if x is None:
            x = self.time_col # default to time column
        if y is None:
            y = self.all_pupil_cols # default to all pupil columns

        if isinstance(y, str):
            y = [y] # make sure y is a list

        # get plot settings
        plot_specific_settings, mpl_kwargs = self._get_plot_settings(x, y, plot_params, is_interactive=False)

        # create subplots with context manager
        with mpl.rc_context(mpl_kwargs):
            fig = plt.figure()
            for i, col in enumerate(y):
                ax = fig.add_subplot(plot_specific_settings['layout'][0], plot_specific_settings['layout'][1], i+1)
                if hue:
                    for trial_group, groupdata in data.groupby(hue, sort=False):

                        # create label for legend
                        label = ', '.join([str(k) for k in trial_group]) if isinstance(trial_group, tuple) else str(trial_group)
                        ax.plot(groupdata[x], groupdata[col], label=label)
                else: # if no hue, plot all data together
                    ax.plot(data[x], data[col])

                # set labels and legend
                ax.set_xlabel(plot_specific_settings['x_title'])
                ax.set_ylabel(plot_specific_settings['y_title'])
                if plot_specific_settings['showlegend']:
                    ax.legend()
                ax.set_title(plot_specific_settings['subplot_titles'][i])

                # configure grid
                ax.grid(plot_specific_settings['grid'])
        
        # save figure if path is provided
        if save:
            plt.savefig(save)

        return fig, ax

    def _plot_trial_interactive(self, trial, x=None, y=None, hue=None, save=None, plot_params=None):
        """
        Create an interactive plot of trial data using Plotly.

        Parameters
        ----------
        trial : pandas.DataFrame
            DataFrame containing trial identifier.
        x : str, optional
            Column name for x-axis values. Defaults to time column.
        y : str or list, optional
            Column name(s) for y-axis values. Defaults to all pupil columns.
        hue : str or list, optional
            Column(s) to group data by for different traces
        save : str, optional
            Path to save the plot
        plot_params : dict, optional
            Dictionary of plot parameters to override defaults
            - layout : tuple of (rows, cols) for subplot layout
            - subplot_titles : list of titles for subplots
            - x_title : str, title of the x-axis
            - y_title : str, title of the y-axis
            - showlegend : bool, whether to show the legend
            - grid : bool, whether to show the grid
            - width : int, width of the plot
            - height : int, height of the plot
            - title_text : str, title of the plot
            - xaxis_showgrid : bool, whether to show the x-axis grid
            - yaxis_showgrid : bool, whether to show the y-axis grid

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure

        Notes
        -----
        - Creates subplots if multiple y variables provided
        - Uses Plotly's default color scheme for traces
        - Allows customization through plot_params dictionary
        """
        if plot_params is None:
            plot_params = {}

        # get data
        data = self.data.copy()

        # get mask
        mask = make_mask(data, trial, invert=True)

        # mask data
        data = data[mask]

        # check if data is empty
        if data.empty:
            trial_info = ', '.join(f"{k}: {v}" for k, v in trial.items())
            raise ValueError(f"No data found for trial with {trial_info}")

        # get x and y
        if x is None:
            x = self.time_col # default to time column
        if y is None:
            y = self.all_pupil_cols # default to all pupil columns

        if isinstance(y, str):
            y = [y] # make sure y is a list

        # get plot settings
        plot_specific_settings, ply_kwargs = self._get_plot_settings(x, y, plot_params, is_interactive=True)

        # plot using plotly
        fig = make_subplots(rows=plot_specific_settings['layout'][0], 
                            cols=plot_specific_settings['layout'][1], 
                            start_cell="top-left",
                            subplot_titles=plot_specific_settings['subplot_titles'],
                            specs =  np.full((plot_specific_settings['layout'][0],plot_specific_settings['layout'][1]), {}).tolist(), # remove margins
                            horizontal_spacing = 0.1, # reduce spacing
                            #vertical_spacing = 0.12
                            )
        
        # default plotly colors
        cols = plotly.colors.DEFAULT_PLOTLY_COLORS

        # iterate over y variables
        for i, col in enumerate(y):
            # figure out row and column
            curr_row = int(i // plot_specific_settings['layout'][1] + 1)
            curr_col = int(i % plot_specific_settings['layout'][1] + 1)
            if hue:
                for g, (trial_group, groupdata) in enumerate(data.groupby(hue, sort=False)):

                    # create label for legend
                    label = ', '.join([str(k) for k in trial_group]) if isinstance(trial_group, tuple) else str(trial_group)
                    # assign color but cycle through colors if more trials than colors
                    curr_color = cols[g % len(cols)]
                    fig.add_trace(go.Scatter(x=groupdata[x], y=groupdata[col], 
                                             mode='lines',
                                             name=label,
                                             line=dict(color=curr_color), 
                                             showlegend=ply_kwargs['showlegend'] if i==0 else False # only show legend for first plot
                                             ), 
                                             row=curr_row, col=curr_col)
            else: 
                curr_color = cols[0]
                fig.add_trace(go.Scatter(x=data[x], y=data[col], 
                                         mode='lines',
                                         name=col,
                                         line=dict(color=curr_color),
                                         showlegend=False
                                         ), 
                                         row=curr_row, col=curr_col)

            # update layout
            fig.update_xaxes(**{k[6:]:v for k, v in ply_kwargs.items() if 'xaxis' in k}) # update x-axis settings
            fig.update_yaxes(**{k[6:]:v for k, v in ply_kwargs.items() if 'yaxis' in k}) # update y-axis settings
            fig.update_layout(**{k:v for k, v in ply_kwargs.items() if 'xaxis' not in k and 'yaxis' not in k}) # update other layout settings
        
        # hack to update font and color for subplot titles
        for i in fig['layout']['annotations']:
            i['text'] = '<b>' + i['text'] + '</b>' # make subplot titles bold
            i['font'] = dict(size=16,color='black') # set font size and color

        # save figure if path is provided
        if save:
            fig.write_image(save)
            
        return fig

    def plot_baseline(self, plot_by=None, show_outliers=True, save=None, interactive=True, plot_params=None, return_fig=False):
        """
        Plot histogram of baseline pupil sizes. This is a wrapper function that calls either 
        plot_baseline_interactive() or plot_baseline_static() depending on the interactive parameter.

        Parameters
        ----------
        plot_by : str or list, optional
            Column(s) to group data by for separate plots.
        show_outliers : bool, default=True
            Whether to show outlier thresholds.
        save : str, optional
            Path to save plot.
        interactive : bool, default=True
            Whether to create interactive plot.
        plot_params : dict, optional
            Additional plotting parameters.
        return_fig : bool, default=False
            Whether to return the figure object.

        Returns
        -------
        figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Plot figure object if return_fig is True.
        axes : matplotlib.axes.Axes, optional
            Plot axes object (only for static plots).

        See Also
        --------
        plot_baseline_interactive : Create interactive baseline histogram plot
        plot_baseline_static : Create static baseline histogram plot
        """
        # plot
        if interactive:
            return self._plot_baseline_interactive(plot_by=plot_by, show_outliers=show_outliers, save=save, plot_params=plot_params, return_fig=return_fig)
        else:
            return self._plot_baseline_static(plot_by=plot_by, show_outliers=show_outliers, save=save, plot_params=plot_params, return_fig=return_fig)
    
    def _plot_baseline_static(self, plot_by=None, show_outliers=True, save=None, plot_params=None, return_fig=False):
        
        """
        Plot histogram of baseline pupil sizes.

        Parameters
        ----------
        plot_by : str or list, optional
            Column(s) to group data by for separate plots.
        show_outliers : bool, default=True
            Whether to show outlier thresholds.
        save : str, optional
            Path to save plot.
        plot_params : dict, default={}
            Additional plotting parameters.
        return_fig : bool, default=False
            Whether to return the figure object.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Plot figure object.
        axes : matplotlib.axes.Axes
            Plot axes object.

        Notes
        -----
        Requires baseline data and optionally baseline outlier information.
        """
        plot_params = plot_params or {}

        # get summary data
        df_summary = self.summary_data.copy()

        # check if baseline data is available
        if ('baseline' not in df_summary.columns):
            raise ValueError("Baseline data is not available. Please run baseline correction first.")
        elif plot_by is not None:
            # convert plot_by to list if not already
            if isinstance(plot_by, str):
                plot_by = [plot_by]
            # check if plot_by columns exist
            if not all(col in df_summary.columns for col in plot_by):
                raise ValueError(f"Plot by column(s) {plot_by} not found in summary data.")
        elif show_outliers and not all(col in df_summary.columns for col in ['baseline_outlier', 'baseline_upper', 'baseline_lower']):
            raise ValueError("Outlier data is not available. Please run check_baseline_outliers first.")

        # check if outlier by is the same as plot_by
        if show_outliers and self.baseline_outlier_by is not None and self.baseline_outlier_by != plot_by: 
            # both outlier by and plot by should be a list at this point
            warnings.warn(f"Outlier detection was performed by {self.baseline_outlier_by}. Plotting by {plot_by}. The plotted thresholds may be incorrect.")
        
        # number of plots
        if plot_by is not None:
            grouped = df_summary.groupby(plot_by, sort=False)
            n_plots = grouped.ngroups
        else:
            grouped = [(None, df_summary)]
            n_plots = 1

        # some additional plot settings specific to histogram plots
        plot_specific_settings = {
            'layout': [(n_plots - 1) // min(2, n_plots) + 1, min(2, n_plots)], # nrows, ncols
            'title': 'Baseline Pupil Sizes',
            'x_title': 'Baseline Pupil Sizes',
            'y_title': 'Count',
            'vline_color': 'red',
            'vline_linestyle': '--',
            'bins': 30,
            'grid': False
        }

        # update plot-specific settings if provided
        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings})

        # update defaults settings if provided
        mpl_kwargs = default_mpl.copy()
        mpl_kwargs['figure.figsize'] = (plot_specific_settings['layout'][1]*8,plot_specific_settings['layout'][0]*3) # ncols, nrows
        mpl_kwargs.update({k:v for k,v in plot_params.items() if k in mpl_kwargs})

        with mpl.rc_context(mpl_kwargs):
                
            # Get unique combinations of grouping variables
            fig, axes = plt.subplots(plot_specific_settings['layout'][0], plot_specific_settings['layout'][1])
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # Plot each group
            for idx, (group_name, group_data) in enumerate(grouped):
                
                # get axis
                ax = axes[idx]
                
                if show_outliers:
                    # Plot histogram
                    sns.histplot(data=group_data, x='baseline', hue='baseline_outlier', 
                            bins=plot_specific_settings['bins'], ax=ax, legend=True)

                    # get thresholds
                    upper_thresh = group_data['baseline_upper'].values[0] # asume all values are the same
                    lower_thresh = group_data['baseline_lower'].values[0] # asume all values are the same

                    # Add threshold lines
                    ax.axvline(upper_thresh, color=plot_specific_settings['vline_color'], 
                            linestyle=plot_specific_settings['vline_linestyle'])
                    ax.axvline(lower_thresh, color=plot_specific_settings['vline_color'], 
                            linestyle=plot_specific_settings['vline_linestyle'])
                    
                    # Add threshold labels
                    ax.text(upper_thresh, ax.get_ylim()[1]*0.1, f'{upper_thresh:.2f}', 
                        rotation=90, va='bottom', ha='right')
                    ax.text(lower_thresh, ax.get_ylim()[1]*0.1, f'{lower_thresh:.2f}', 
                        rotation=90, va='bottom', ha='right')
                else:
                    # Plot histogram
                    sns.histplot(data=group_data, x='baseline', ax=ax, bins=plot_specific_settings['bins'])

                # Set labels
                ax.set_xlabel(plot_specific_settings['x_title'])
                ax.set_ylabel(plot_specific_settings['y_title'])
                
                # Set title
                if n_plots > 1:
                    if isinstance(group_name, tuple):
                        title = ' | '.join([x for x in group_name])
                    else:
                        title = f'{group_name}'
                    ax.set_title(title)
                    fig.suptitle(plot_specific_settings['title'])
                else:
                    ax.set_title(plot_specific_settings['title'])
                
                # Configure grid
                ax.grid(plot_specific_settings['grid'])
            
            # Remove empty subplots if any
            for idx in range(n_plots, len(axes)):
                fig.delaxes(axes[idx])
            
            fig.tight_layout()

            # Save figure if path is provided
            if save:
                plt.savefig(save, bbox_inches='tight', dpi=mpl_kwargs['figure.dpi'])

            # return figure 
            if return_fig:
                return fig, axes

    def _plot_baseline_interactive(self, plot_by=None, show_outliers=True, save=None, plot_params=None, return_fig=True):
        
        """
        Create interactive histogram plot of baseline pupil sizes using Plotly.

        Parameters
        ----------
        plot_by : str or list, optional
            Column(s) to group data by for separate plots.
        show_outliers : bool, default=True
            Whether to show outlier thresholds.
        save : str, optional
            Path to save plot.
        plot_params : dict, default={}
            Additional plotting parameters.
        return_fig : bool, default=True
            Whether to return the figure object.
        
        Returns
        -------
        figure : plotly.graph_objects.Figure
            Interactive Plotly figure object.

        Notes
        -----
        Requires baseline data and optionally baseline outlier information.
        """
        plot_params = plot_params or {}

        # get summary data
        df_summary = self.summary_data.copy()

        # check if baseline data is available
        if ('baseline' not in df_summary.columns):
            raise ValueError("Baseline data is not available. Please run baseline correction first.")
        elif plot_by is not None:
            # convert plot_by to list if not already
            if isinstance(plot_by, str):
                plot_by = [plot_by]
            # check if plot_by columns exist
            if not all(col in df_summary.columns for col in plot_by):
                raise ValueError(f"Plot by column(s) {plot_by} not found in summary data.")
        elif show_outliers and not all(col in df_summary.columns for col in ['baseline_outlier', 'baseline_upper', 'baseline_lower']):
            raise ValueError("Outlier data is not available. Please run check_baseline_outliers first.")

        # check if outlier by is the same as plot_by
        if show_outliers and self.baseline_outlier_by is not None and self.baseline_outlier_by != plot_by: 
            # both outlier by and plot by should be a list at this point
            warnings.warn(f"Outlier detection was performed by {self.baseline_outlier_by}. Plotting by {plot_by}. The plotted thresholds may be incorrect.")
        
        # Get groups
        if plot_by is not None:
            grouped = df_summary.groupby(plot_by, sort=False)
        else:
            grouped = [(None, df_summary)]

        # Plot settings
        plot_specific_settings = {
            'title': 'Baseline Pupil Sizes',
            'x_title': 'Baseline Pupil Sizes',
            'y_title': 'Count',
            'vline_color': 'red',
            'vline_style': 'dash',
            'bins': 30,
            'grid': False,
            'bargap': 0,  # Remove gap between bars
            'bargroupgap': 0  # Remove gap between bar groups
        }

        # Update plot settings if provided
        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings})

        # Update Plotly defaults if provided
        ply_kwargs = default_plotly.copy()
        ply_kwargs['width'] = plot_params.get('width', 800)
        ply_kwargs['height'] = plot_params.get('height', 500)
        ply_kwargs['title_text'] = plot_params.get('title_text', plot_specific_settings['title'])
        ply_kwargs['xaxis_title_text'] = plot_params.get('xaxis_title_text', plot_specific_settings['x_title'])
        ply_kwargs['yaxis_title_text'] = plot_params.get('yaxis_title_text', plot_specific_settings['y_title'])
        ply_kwargs['xaxis_showgrid'] = plot_params.get('xaxis_showgrid', plot_specific_settings['grid'])
        ply_kwargs['yaxis_showgrid'] = plot_params.get('yaxis_showgrid', plot_specific_settings['grid'])
        ply_kwargs['bargap'] = plot_specific_settings['bargap']
        ply_kwargs['bargroupgap'] = plot_specific_settings['bargroupgap']
        ply_kwargs.update({k:v for k,v in plot_params.items() if k in ply_kwargs})

        # Create figure
        fig = go.Figure()

        # Create dropdown menu options
        dropdown_options = []
        
        # Keep track of all traces for each group
        all_traces_in_groups = []
        
        # Add traces for each group
        for groupid, (group_name, group_data) in enumerate(grouped):
            # Format group name for display
            group_title = ' | '.join([str(x) for x in group_name]) if isinstance(group_name, tuple) else str(group_name) if group_name is not None else "All"
            
            # Keep track of number of traces for this group
            traces_in_group = []
            
            if show_outliers:
                # Add histogram trace for non-outliers
                non_outlier_data = group_data[~group_data['baseline_outlier']]['baseline'].dropna()
                if len(non_outlier_data) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=non_outlier_data,
                            name='Non-outliers',
                            nbinsx=plot_specific_settings['bins'],
                            visible=(groupid == 0),
                            showlegend=True,
                            opacity=0.7,
                            marker_color='#4C72B0',  # seaborn default blue
                            marker_line_color='black',
                            marker_line_width=2
                        )
                    )
                    traces_in_group.append(len(fig.data) - 1)

                # Add histogram trace for outliers
                outlier_data = group_data[group_data['baseline_outlier']]['baseline'].dropna()
                if len(outlier_data) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=outlier_data,
                            name='Outliers',
                            nbinsx=plot_specific_settings['bins'],
                            visible=(groupid == 0),
                            showlegend=True,
                            opacity=0.7,
                            marker_color='#DD8452',  # seaborn default orange
                            marker_line_color='black',
                            marker_line_width=2
                        )
                    )
                    traces_in_group.append(len(fig.data) - 1)

                # Add threshold lines
                upper_thresh = group_data['baseline_upper'].values[0]
                lower_thresh = group_data['baseline_lower'].values[0]

                # Add upper threshold line
                fig.add_trace(
                    go.Scatter(
                        x=[upper_thresh, upper_thresh],
                        y=[0, max(non_outlier_data)],  # Use pre-calculated max_y
                        mode='lines',
                        name=f'Upper threshold: {upper_thresh:.2f}',
                        line=dict(
                            color=plot_specific_settings['vline_color'],
                            dash=plot_specific_settings['vline_style'],
                            width=2
                        ),
                        visible=(groupid == 0),
                        showlegend=True,
                        hovertemplate=f"Upper threshold: {upper_thresh:.2f}"
                    )
                )
                traces_in_group.append(len(fig.data) - 1)

                # Add lower threshold line
                fig.add_trace(
                    go.Scatter(
                        x=[lower_thresh, lower_thresh],
                        y=[0, max(non_outlier_data)],  # Use pre-calculated max_y
                        mode='lines',
                        name=f'Lower threshold: {lower_thresh:.2f}',
                        line=dict(
                            color=plot_specific_settings['vline_color'],
                            dash=plot_specific_settings['vline_style'],
                            width=2
                        ),
                        visible=(groupid == 0),
                        showlegend=True,
                        hovertemplate=f"Lower threshold: {lower_thresh:.2f}"
                    )
                )
                traces_in_group.append(len(fig.data) - 1)

            else:
                # Add single histogram trace without outlier distinction
                fig.add_trace(
                    go.Histogram(
                        x=group_data['baseline'].dropna(),
                        name='All trials',
                        nbinsx=plot_specific_settings['bins'],
                        visible=(groupid == 0),
                        showlegend=False,
                        opacity=0.7,
                        marker_color='lightblue',
                        marker_line_color='black',
                        marker_line_width=2
                    )
                )
                traces_in_group.append(len(fig.data) - 1)

            # Store traces for this group
            all_traces_in_groups.append({
                'traces': traces_in_group,
                'title': group_title
            })

        # Now create visibility settings outside the loop
        total_traces = len(fig.data)
        for group_info in all_traces_in_groups:
            vis = [False] * total_traces
            for trace_idx in group_info['traces']:
                vis[trace_idx] = True
            
            # Add dropdown option
            dropdown_options.append(
                dict(
                    args=[
                        {"visible": vis},
                        {
                            "title": f"{plot_specific_settings['title']} - {group_info['title']}",
                            "showlegend": True
                        }
                    ],
                    label=group_info['title'],
                    method="update"
                )
            )

        # Update layout
        fig.update_layout(
            updatemenus=[{
                'buttons': dropdown_options,
                'direction': 'down',
                'showactive': True,
                'x': 1.2,
                'y': 1.2,
                'xanchor': 'right',
                'yanchor': 'top'
            }],
            barmode='overlay',  # Overlay histograms
            **ply_kwargs
        )

        # Save if requested
        if save:
            if save.endswith('.html'):
                fig.write_html(save)
            else:
                raise ValueError(f"Interactive plots must be saved as html file. Got {save}.")

        # return figure 
        if return_fig:
            return fig
        else:
            display(fig)

    def plot_spaghetti(self, x=None, y=None, show_outliers=True, plot_by=None, save=False, plot_params=None, return_fig=True): 
        """
        Plot pupil traces for all trials as a spaghetti plot.

        Parameters
        ----------
        x : str, optional
            Column name for x-axis. Defaults to time column.
        y : str, optional 
            Column name for y-axis. Defaults to latest pupil column.
        show_outliers : bool, default=True
            Whether to highlight outlier traces.
        plot_by : str or list, optional
            Column(s) to group data by for separate plots.
        save : str, optional
            Path to save plot. Only supports html files. If None, plot is not saved.
        plot_params : dict, default={}
            Additional plotting parameters.
        return_fig : bool, default=True
            Whether to return the figure object.

        Returns
        -------
        plotly.graph_objects.Figure
            Plot figure object if return_fig is True.

        Notes
        -----
        Creates an interactive spaghetti plot showing pupil traces for all trials.
        If plot_by is specified, creates separate subplots for each group using dropdown menus.
        Outlier traces can be highlighted if outlier detection was performed.
        """
        plot_params = plot_params or {}
        
        # get summary data 
        df_summary = self.summary_data.copy()

        # check if trace_outlier is in summary_data
        if show_outliers and ('trace_outlier' not in df_summary.columns):
            raise ValueError("trace_outlier column not found in summary_data. Please run check_trace_outliers first.")

        # get x and y
        if x is None:
            x = self.time_col # default to time column
        if y is None:
            y = self.all_pupil_cols[-1] # default to last pupil column

        # get data
        df_plot = self.data.copy()
        if plot_by is not None:
            # convert plot_by to list if not already
            if isinstance(plot_by, str):
                plot_by = [plot_by]
            # get unique columns
            cols = [x, y] + plot_by + self.trial_identifier 
            cols = list(set(cols))
            grouped = df_plot[cols].groupby(plot_by, sort=False)
        else:
            cols = [x, y] + self.trial_identifier
            cols = list(set(cols))
            grouped = [(None, df_plot[cols])]
        
        # check if outlier by is the same as plot_by
        if show_outliers and self.trace_outlier_by is not None and self.trace_outlier_by != plot_by: 
            # both outlier by and plot by should be a list at this point
            warnings.warn(f"Outlier detection was performed by {self.trace_outlier_by}. Plotting by {plot_by}. The plotted thresholds may be incorrect.")

        # plot
        # some additional plot settings specific to this plot
        plot_specific_settings = {
            'title': 'Spaghetti plot',
            'subplot_titles': [' | '.join([str(x) for x in group]) for group, _ in grouped] if plot_by is not None else None,
            'x_title': x,
            'y_title': y,
            # line settings
            'line_width': 2,
            'line_style': 'solid',
            # hline settings
            'hline_color': 'black',
            'hline_style': 'dash',
            'hline_width': 2,
            # grid
            'grid': False
            }

        # update plot-specific settings if provided
        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings.keys()}) # keep only additional plot-specific keys

        # update defaults settings if provided
        ply_kwargs = default_plotly.copy()
        ply_kwargs['width'] = plot_params.get('width', 1200)
        ply_kwargs['height'] = plot_params.get('height', 400)
        ply_kwargs['title_text'] = plot_params.get('title_text', plot_specific_settings['title']) # override default title
        ply_kwargs['xaxis_title_text'] = plot_params.get('xaxis_title_text', plot_specific_settings['x_title']) # override default x-axis title
        ply_kwargs['yaxis_title_text'] = plot_params.get('yaxis_title_text', plot_specific_settings['y_title']) # override default y-axis title
        ply_kwargs['xaxis_showgrid'] = plot_params.get('xaxis_showgrid', plot_specific_settings['grid']) # override default x-axis grid
        ply_kwargs['yaxis_showgrid'] = plot_params.get('yaxis_showgrid', plot_specific_settings['grid']) # override default y-axis grid
        ply_kwargs.update({k:v for k,v in plot_params.items() if k in ply_kwargs})

        # plot using plotly
        fig = go.Figure()

        # Create a list to store visibility settings for each trace
        all_traces = []
        visible_settings = []
        dropdown_options = []
        
        # Add traces for each group
        for groupid, (group, groupdata) in enumerate(grouped):
            traces_in_group = []
            dropdown_options.append({
                'label': plot_specific_settings['subplot_titles'][groupid] if group is not None else "All",
                'method': "update",
                'args': [{"visible": []}, {"title": ""}]  # Will be filled later
            })
            
            # Add traces for each trial in the group
            for trial_id, (trial, trialdata) in enumerate(groupdata.groupby(self.trial_identifier, sort=False)):
                is_outlier = False
                if show_outliers:
                    is_outlier = df_summary.loc[np.all(df_summary[self.trial_identifier] == trial, axis=1), 'trace_outlier'].values[0]
                
                alpha = 1 if is_outlier or not show_outliers else 0.2
                showlegend = True if is_outlier else False
                label = ', '.join([f"{k}: {v}" for k,v in zip(self.trial_identifier, trial)]) if is_outlier else None
                
                # downsample trialdata by selecting every 10th sample for faster plotting
                downsample_mask = np.arange(len(trialdata)) % 10 == 0
                downsampled = trialdata[downsample_mask]

                trace = go.Scatter(
                    x=downsampled[x],
                    y=downsampled[y],
                    name=label,
                    mode='lines',
                    line=dict(width=plot_specific_settings['line_width']),
                    line_dash=plot_specific_settings['line_style'],
                    opacity=alpha,
                    showlegend=showlegend,
                    visible=(groupid == 0),  # Only first group visible initially
                    hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>" +
                    "<br>".join([f"{k}: {v}" for k,v in dict(zip(self.trial_identifier, trial)).items()]) +
                    "<extra></extra>"
                )
                fig.add_trace(trace)
                traces_in_group.append(trace)
                
            # Add threshold lines if showing outliers
            if show_outliers:
                # Get thresholds from the first trial in the group
                first_trial = next(iter(groupdata.groupby(self.trial_identifier, sort=False)))[0]
                upper_threshold = df_summary.loc[np.all(df_summary[self.trial_identifier] == first_trial, axis=1), 'trace_upper'].values[0]
                lower_threshold = df_summary.loc[np.all(df_summary[self.trial_identifier] == first_trial, axis=1), 'trace_lower'].values[0]
                
                # Add upper threshold line
                trace_upper = go.Scatter(
                    x=[downsampled[x].min(), downsampled[x].max()],
                    y=[upper_threshold, upper_threshold],
                    mode='lines',
                    line=dict(dash=plot_specific_settings['hline_style'],
                            color=plot_specific_settings['hline_color'],
                            width=plot_specific_settings['hline_width']),
                    name=f'Upper threshold: {upper_threshold:.2f}',
                    showlegend=False,
                    visible=(groupid == 0)
                )
                fig.add_trace(trace_upper)
                traces_in_group.append(trace_upper)
                
                # Add lower threshold line
                trace_lower = go.Scatter(
                    x=[downsampled[x].min(), downsampled[x].max()],
                    y=[lower_threshold, lower_threshold],
                    mode='lines',
                    line=dict(dash=plot_specific_settings['hline_style'],
                            color=plot_specific_settings['hline_color'],
                            width=plot_specific_settings['hline_width']),
                    name=f'Lower threshold: {lower_threshold:.2f}',
                    showlegend=False,
                    visible=(groupid == 0)
                )
                fig.add_trace(trace_lower)
                traces_in_group.append(trace_lower)

            all_traces.append(traces_in_group)
            
        # Create visibility settings for each dropdown option
        for i in range(len(all_traces)):
            vis = []
            for j, traces in enumerate(all_traces):
                vis.extend([True if j == i else False] * len(traces))
            visible_settings.append(vis)
            
            # Update the args for each dropdown option
            dropdown_options[i]['args'][0]["visible"] = vis
            dropdown_options[i]['args'][1]["title"] = f"{plot_specific_settings['title']} - {dropdown_options[i]['label']}"
        
        # Update layout to include dropdown menu
        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                x=1.0,  # Position the dropdown at the right
                y=1.2,  # Position slightly above the plot
                showactive=True,
                active=0,  # Show first group by default
                buttons=dropdown_options
            )]
            )
  
        # Update layout
        fig.update_xaxes(**{k[6:]:v for k, v in ply_kwargs.items() if 'xaxis' in k})
        fig.update_yaxes(**{k[6:]:v for k, v in ply_kwargs.items() if 'yaxis' in k})
        fig.update_layout(**{k:v for k, v in ply_kwargs.items() if 'xaxis' not in k and 'yaxis' not in k})

        # Save figure if path is provided
        if save:
            fig.write_html(save)

        # return figure if requested
        if return_fig:
            return fig
        else:
            display(fig)


    def plot_evoked(self, data=None, time_col=None, pupil_col=None, condition=None, error='ci', save=None, plot_params=None, **kwargs):
        """
        Plot evoked pupil response.

        Creates plot of average pupil response across trials, optionally split by condition.

        Parameters
        ----------
        data : str or pandas.DataFrame, optional
            Data to plot. If string, uses corresponding attribute.
        time_col : str, optional
            Column name for time values.
        pupil_col : str, optional
            Column name for pupil values.
        condition : str or list, optional
            Column(s) to split data by.
        error : {'ci', 'sem', 'std', None}, default='ci'
            Type of error to plot:
            - 'ci': bootstrap confidence interval
            - 'sem': standard error of the mean
            - 'std': standard deviation
            - None: no error bars
        save : str, optional
            Path to save plot.
        plot_params : dict, default={}
            Additional plotting parameters.
        **kwargs
            Additional arguments passed to confidence interval calculation.

        Returns
        -------
        arrays_by_condition : dict
            Dictionary of arrays containing trial data for each condition.
        (figure, axes) : tuple
            Plot figure and axes objects.
        """
        plot_params = plot_params or {}
        
        # get data
        if data is None:
            data = self.data.copy()
        else:
            data = getattr(self, data)

        # get samp_freq
        samp_freq = self.samp_freq

        # get column
        if pupil_col is None:
            pupil_col = self.all_pupil_cols[-1]

        # handle condition
        if condition is not None:
            if isinstance(condition, str):
                condition = [condition]
            # get unique values for each condition
            condition_values = {cond: data[cond].unique() for cond in condition}

        # if no condition, process all data together
        if condition is None:
            grouped = data.groupby(self.trial_identifier, sort=False)
            ngroup = grouped.ngroups
            min_len = grouped[pupil_col].count().min()
            print(f'{ngroup} trials. Minimum number of samples: {min_len}. Data will be padded to size: {ngroup} x {min_len}')

            # initialize empty array
            test_array = np.empty((ngroup, min_len))

            # iterate over groups
            for i, (group, groupdata) in enumerate(grouped):
                vals = np.asarray(groupdata[pupil_col].to_list())
                vals = vals[:min_len]
                test_array[i,:] = vals

            arrays_by_condition = {'all': test_array}
            
        else:
            # create all combinations of condition values
            import itertools
            condition_combinations = list(itertools.product(*[condition_values[cond] for cond in condition]))
            arrays_by_condition = {}
            
            for comb in condition_combinations:
                # create mask for this combination
                mask = pd.Series(True, index=data.index)
                for cond, val in zip(condition, comb):
                    mask &= (data[cond] == val)
                
                # get data for this combination
                subset = data[mask]
                grouped = subset.groupby(self.trial_identifier, sort=False)
                ngroup = grouped.ngroups
                if ngroup == 0:
                    continue
                    
                min_len = grouped[pupil_col].count().min()
                print(f'Condition {dict(zip(condition, comb))}: {ngroup} trials. Minimum samples: {min_len}')
                
                # initialize empty array
                test_array = np.empty((ngroup, min_len))
                
                # iterate over groups
                for i, (group, groupdata) in enumerate(grouped):
                    vals = np.asarray(groupdata[pupil_col].to_list())
                    vals = vals[:min_len]
                    test_array[i,:] = vals
                    
                # store array with condition name
                cond_name = '_'.join([f'{c}_{v}' for c,v in zip(condition, comb)])
                arrays_by_condition[cond_name] = test_array

        # plot settings
        plot_specific_settings = {
        'title': 'Evoked Pupil Size',
        'x_title': 'Time since stimulus onset(s)',
        'y_title': 'Pupil Size',
        'vline_color': 'red',
        'vline_linestyle': '--',
        'grid': False,
        'legend_labels': list(arrays_by_condition.keys())
        }

        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings})
        mpl_kwargs = default_mpl.copy()
        mpl_kwargs.update({k:v for k,v in plot_params.items() if k in mpl_kwargs})

        # create plot with context manager
        with mpl.rc_context(mpl_kwargs):
            fig, ax = plt.subplots()
            
            for i, (cond_name, test_array) in enumerate(arrays_by_condition.items()):

                # get time array 
                t = np.arange(test_array.shape[1]) / samp_freq
                
                if error == 'ci':
                    try:
                        import mne.stats as ms
                        ci_low, ci_high = ms.bootstrap_confidence_interval(test_array, **kwargs)
                    except ImportError:
                        warnings.warn("mne is not installed. Not computing confidence interval.")
                        ci_low, ci_high = None, None
                elif error == 'sem':
                    ci_low = test_array.mean(axis=0) - test_array.std(axis=0) / np.sqrt(test_array.shape[0])
                    ci_high = test_array.mean(axis=0) + test_array.std(axis=0) / np.sqrt(test_array.shape[0])
                elif error == 'std':
                    ci_low = test_array.mean(axis=0) - test_array.std(axis=0)
                    ci_high = test_array.mean(axis=0) + test_array.std(axis=0)
                else:
                    ci_low, ci_high = None, None

                ax.plot(t, test_array.mean(axis=0), label=plot_specific_settings['legend_labels'][i])
                if error and ci_low is not None and ci_high is not None:
                    ax.fill_between(t, ci_low, ci_high, alpha=0.2)

            if len(arrays_by_condition) > 1:
                ax.legend()
                
            ax.set_xlabel(plot_specific_settings['x_title'])
            ax.set_ylabel(plot_specific_settings['y_title'])
            ax.set_title(plot_specific_settings['title'])
            ax.grid(plot_specific_settings['grid'])

        if save:
            plt.savefig(save, bbox_inches='tight', dpi=mpl_kwargs['figure.dpi'])

        return arrays_by_condition, (fig, ax)


    def save(self, path):
        """
        Save object to file using dill.

        Parameters
        ----------
        path : str
            Path to save file.
        """

        # check if dill is installed
        try:
            import dill
        except ImportError:
            raise ImportError("dill is not installed. Please install dill using pip install dill.") 
        
        # save data
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load object from file using dill.

        Parameters
        ----------
        path : str
            Path to load file from.

        Returns
        -------
        object
            Loaded object.
        """

        # check if dill is installed
        try:
            import dill
        except ImportError:
            raise ImportError("dill is not installed. Please install dill using pip install dill.") 

        # load data
        with open(path, 'rb') as f:
            return dill.load(f) 

    def copy(self):
        """
        Create a deep copy of the object.

        Returns
        -------
        object
            Deep copy of self.
        """
        import copy
        # deepcopy
        return copy.deepcopy(self)

def compute_speed(x, y):
    """
    Compute the speed of change between two arrays.

    Takes two arrays x and y and computes the rate of change (speed) between corresponding points.
    The speed is calculated as the absolute maximum of the forward and backward differences at each point.

    Parameters
    ----------
    x : array-like
        First array of values
    y : array-like 
        Second array of values, must be same length as x

    Returns
    -------
    numpy.ndarray
        Array of speed values with same length as input arrays. Contains NaN values at endpoints
        and where division by zero or invalid values occur.

    Notes
    -----
    - Uses np.diff() to compute differences between consecutive points
    - Takes absolute maximum of forward/backward differences at each point
    - Suppresses RuntimeWarnings for NaN/inf values
    - Sets NaN/inf values to NaN in output
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = np.diff(x) / np.diff(y)
    speed_diff = np.abs(np.column_stack((np.insert(diff, 0, np.nan), np.append(diff, np.nan))))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed_diff = np.nanmax(speed_diff, axis=1)
        speed_diff[np.isnan(speed_diff) | np.isinf(speed_diff)] = np.nan
        
    return speed_diff

def convert_pupil(pupil_size, artificial_d, artificial_size, recording_unit='diameter'):
    """
    Convert pupil measurements between different recording units.
    
    Converts pupil measurements from raw units to millimeters using calibration values.
    Handles both diameter and area measurements.

    Parameters
    ----------
    pupil_size : float or array-like
        Pupil size in recording units (diameter or area)
    artificial_d : float
        Diameter of artificial pupil used for calibration (in mm)
    artificial_size : float
        Size of artificial pupil in recording units (diameter or area)
    recording_unit : str, optional (default='diameter')
        Unit of the recorded measurements - either 'diameter' or 'area'

    Returns
    -------
    numpy.ndarray
        Converted pupil measurements in millimeters

    Notes
    -----
    - The unit of artifiical_size must be the same as the unit of the actual recording (either diameter or area).
    - The unit of artificial_d is always in mm.
    - For diameter recordings, applies linear scaling
    - For area recordings, takes square root before scaling
    - Raises ValueError if recording_unit is invalid
    """
    if recording_unit == 'diameter':
        return artificial_d * pupil_size / artificial_size
    elif recording_unit == 'area':
        return artificial_d * np.sqrt(pupil_size) / np.sqrt(artificial_size)
    else:
        raise ValueError(f"Invalid recording unit: {recording_unit}")