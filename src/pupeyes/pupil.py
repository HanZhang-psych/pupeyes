# -*- coding:utf-8 -*-

"""
Pupil Data Processing Module

This module provides tools for processing pupillometry data from eye trackers.
It includes functionality for deblinking, smoothing, baseline correction, and plotting
pupil size data.
"""

import warnings
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
import dill
from .utils import make_mask, lowpass_filter
from .aoi import is_inside
from .external.based_noise_blinks_detection import based_noise_blinks_detection

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .defaults import default_mpl, default_plotly

class PupilProcessor:
    """
    A class for processing and analyzing pupillometry data.

    This class provides methods for preprocessing pupil size data, including blink removal,
    artifact rejection, smoothing, and baseline correction. It also includes tools for
    data visualization and analysis.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing pupil size data and associated measurements
    trial_identifier : str or list
        Column name(s) identifying unique trials. If list, trials are uniquely
        identified by the combination of these columns.
    pupil_col : str
        Column name containing pupil size measurements
    time_col : str
        Column name containing timestamps. Must be in milliseconds and integer.
    x_col : str
        Column name containing x-coordinates of gaze position
    y_col : str
        Column name containing y-coordinates of gaze position
    samp_freq : float
        Sampling frequency of the eye tracker in Hz
    convert_pupil_size : bool, default=False
        Whether to convert pupil size from area to diameter or vice versa
    artificial_d : float, default=5
        Artificial pupil diameter in mm, used for pupil size conversion
    artificial_size : float, default=5663
        Artificial pupil size in arbitrary units, used for pupil size conversion
    recording_unit : {'diameter', 'area'}, default='diameter'
        Unit of the recorded pupil size
    device : {'eyelink', 'tobii_titta', 'tobii_prolab','smi'}, default='eyelink'
        Device type. At the moment, this only controls whether sampling frequency is checked.
    eyetracker_missing_value : int, default=0
        Value for missing pupil size for the eye tracker. Different eye trackers use different values to indicate missing values. PupEyes will replace these values with 0.
        Other possible values are pd.NA, np.nan, -1, -999, etc.
    progress_bar : bool, default=True
        Whether to show a progress bar for preprocessing steps

    Attributes
    ----------
    data : pd.DataFrame
        The processed pupil data
    summary_data : pd.DataFrame
        Summary statistics for each trial
    trials : pd.DataFrame
        A dataframe of unique trial identifiers
    params : dict
        Dictionary storing parameters used in processing steps
    all_pupil_cols : list
        List of column names containing pupil data at different processing stages
    all_steps : list
        List of processing steps applied to the data

    Notes
    -----
    - All processing methods return self for method chaining
    - Most methods create new columns with processed data rather than modifying existing ones
    - Processing parameters are stored in the params dictionary for reproducibility
    - Summary statistics are automatically updated after each processing step
    - artificial_d is the diameter of an artificial pupil provided by Eyelink. 
    - artificial_size was measured for the setup of our research group and may not generalize to other setups.
    """

    def __init__(self, data, trial_identifier, pupil_col, time_col, x_col, y_col, samp_freq, convert_pupil_size=False, artificial_d=5, artificial_size=5663, recording_unit='diameter', device='eyelink', eyetracker_missing_value=0, progress_bar=True):
        """
        Initialize PupilProcessor object.
        """
        #### device ####
        self.device = device
        print(f'Device: {self.device}')

        #### data ####
        # x gaze position
        self.x_col = x_col 
        # y gaze position
        self.y_col = y_col
        # time column
        self.time_col = time_col 
        # pupil column
        self.pupil_col = pupil_col 
        # trial identifier
        # group by column for preprocessing
        if isinstance(trial_identifier, str):
            self.trial_identifier = [trial_identifier]
        else:
            self.trial_identifier = trial_identifier

        # make a copy of the data
        self.data = data.copy().convert_dtypes()

        #### handle missing values ####
        self.eyetracker_missing_value = eyetracker_missing_value
        # replace eye-tracker specified missing values with 0
        # check if the missing value exists in the data
        if pd.isna(self.eyetracker_missing_value):
            # handle pd.NA or np.nan
            if self.data[self.pupil_col].isna().any():
                print(f'Eye-tracker missing value is {self.eyetracker_missing_value}. Replacing with 0.')
                # convert to numeric to handle mixed types, then replace NA values
                self.data[self.pupil_col] = pd.to_numeric(self.data[self.pupil_col], errors='coerce').fillna(0)
        else:
            # handle specific numeric values
            if (self.data[self.pupil_col] == self.eyetracker_missing_value).any():
                if self.eyetracker_missing_value == 0:
                    print(f'Eye-tracker missing value for pupil size is 0. No replacement needed.')
                else:
                    print(f'Eye-tracker missing value is {self.eyetracker_missing_value}. Replacing with 0.')
                    self.data[self.pupil_col] = self.data[self.pupil_col].replace({self.eyetracker_missing_value: 0})

        # check for non-integer timestamps and warn
        time_values = self.data[time_col].dropna()
        if not all(isinstance(val, (int, np.integer)) for val in time_values):
            import warnings
            warnings.warn(
                f"Non-integer timestamps detected in column '{time_col}'. "
                "The preprocessing pipeline expects integer timestamps in milliseconds. "
                "Decimal timestamps may cause issues in some preprocessing steps.",
                UserWarning
            )
        
        #### check sampling frequency ####
        # sampling frequency
        self.samp_freq = samp_freq
        # check sampling frequency
        self.check_sampling_frequency()

        #### convert pupil size ####
        if convert_pupil_size:
            self.recording_unit = recording_unit
            self.artificial_d = artificial_d
            self.artificial_size = artificial_size
            self.data[self.pupil_col] = convert_pupil(self.data[self.pupil_col], artificial_d=artificial_d, artificial_size=artificial_size, recording_unit=recording_unit)
            print(f'Pupil data converted to {recording_unit} with artificial d={artificial_d} and artificial size={artificial_size}')
        else:
            self.recording_unit = None
            self.artificial_d = None
            self.artificial_size = None

        #### other stuff ####
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
        # progress bar
        self.progress_bar = progress_bar

        print(f'PupilProcessor initialized with {len(self.data)} samples')
        print(f'Pupil column: {self.pupil_col}, Time column: {self.time_col}, X column: {self.x_col}, Y column: {self.y_col}')
        print(f'Trial identifier: {self.trial_identifier}, Number of trials: {len(self.trials)}')

    def check_sampling_frequency(self, sampling_rate=None, data=None):
        """
        Check if the sampling frequency is consistent. Only performed for Eyelink data.

        This method checks if the sampling frequency is consistent across trials.
        If not, it raises an error. It is automatically called when initializing the PupilProcessor.
        If resampling is performed, the sampling frequency is checked again.
        
        Parameters
        ----------
        sampling_rate : int, default=None
            Sampling rate to check. If None, the sampling rate is checked against the current sampling rate.
        data : pd.DataFrame, default=None
            Data to check. If None, the data is checked against the current data. The time column must be in milliseconds and integer.

        Returns
        -------
        check_pass : bool
            True if the sampling frequency is consistent, False otherwise
        """
        if self.device not in ['eyelink']:
            print(f'Sampling frequency check skipped for {self.device} data.')
            return True
        
        data = self.data if data is None else data
        sampling_rate = self.samp_freq if sampling_rate is None else sampling_rate
        check_pass = False
        # check if the difference between consecutive samples is equal to a fixed value
        diff = data.groupby(self.trial_identifier, sort=False)[self.time_col].diff().dropna().unique()
        if len(diff) == 1:
            if 1000/diff[0] != sampling_rate:
                raise ValueError(f'Actual sampling frequency {1000/diff[0]}Hz does not match the provided sampling frequency {sampling_rate}Hz!')
            else:
                print(f'Sampling frequency check passed. Sampling rate: {sampling_rate}Hz')
                check_pass = True
        else:
            raise ValueError('Sampling frequency is not consistent!')
        
        return check_pass

    def deblink(self, suffix='_db'):
        """
        Remove blinks from pupil data using noise-based blink detection.

        This method identifies and removes blinks from pupil data using the based_noise_blinks_detection 
        algorithm. Blinks are detected based on rapid changes in pupil size characteristic of eye 
        closure. The method processes each trial separately and creates a new column containing the 
        deblinked data.

        Parameters
        ----------
        suffix : str, default='_db'
            Suffix to append to the pupil column name for the deblinked data.
            For example, if pupil column is 'pupil', the new column will be 'pupil_db'.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with:
            - run_deblink: Boolean indicating if deblinking was performed
            - pct_deblink: Percentage of samples identified as blinks
        - Creates a new column with suffix appended to the current pupil column name
        - Updates all_pupil_cols and all_steps to track processing history
        - Blink periods are replaced with NaN values
        - Trials with all missing pupil data are skipped and reported
        - Processing parameters are stored in self.params['deblink']

        See Also
        --------
        based_noise_blinks_detection : The underlying blink detection algorithm
        """
        # store parameters
        self.params['deblink'] = {k:v for k,v in locals().items() if k != 'self'}

        # create new column for deblinked data
        pupil_col = self.all_pupil_cols[-1]
        new_col = pupil_col + suffix
        self.data[new_col] = self.data[pupil_col] # default to last pupil column

        # get sampling frequency
        samp_freq = self.samp_freq
        
        print(f'Running deblink using sampling frequency {samp_freq}Hz')

        # initialize blinks removed column in summary data
        self.summary_data['run_deblink'] = False
        self.summary_data['pct_deblink'] = pd.NA

        # iterate over trials if trial_identifier is provided
        empty_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Deblinking', disable=not self.progress_bar):
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
                #nblink = len(blinks["blink_onset"])
                nblinksamps = int(np.sum(np.array(blinks["blink_offset"]) - np.array(blinks["blink_onset"])))
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_deblink'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_deblink'] = nblinksamps/len(groupdata)

        # replace potential other 0 values with NaN
        self.data[new_col] = self.data[new_col].replace({0:pd.NA})

        # update latest pupil column 
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Deblinked')

        # print summary
        print(f"✓ Deblinking completed!")
        print(f"  → New column: '{new_col}' (blinks removed)")
        print(f"  → Previous column '{pupil_col}' preserved.")
        print(f"  → {len(empty_trials)} trial(s) failed.")

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self

    def artifact_rejection(self, suffix='_ar', method='both', speed_n=16, zscore_threshold=2.5, zscore_allowp=0.1):
        """
        Reject artifacts from pupil data using speed and/or z-score based methods.

        This method identifies and removes artifacts using two possible approaches:
        1. Speed-based: Removes samples where pupil size changes too rapidly
        2. Z-score based: Removes extreme values based on z-score thresholds
        
        The method can use either approach individually or combine both.

        Parameters
        ----------
        suffix : str, default='_ar'
            Suffix to append to the pupil column name for the artifact-rejected data.
            For example, if pupil column is 'pupil', the new column will be 'pupil_ar'.
        method : {'speed', 'zscore', 'both'}, default='both'
            Method to use for artifact rejection:
            - 'speed': Use only speed-based rejection
            - 'zscore': Use only z-score based rejection
            - 'both': Use both methods
        speed_n : int, default=16
            Number of MADs above median speed to use as threshold for speed-based rejection
        zscore_threshold : float, default=2.5
            Z-score threshold for artifact rejection for z-score based rejection
        zscore_allowp : float, default=0.1
            Proportion of mean to use as minimum standard deviation for z-score based rejection. 
            If sd/mean < zscore_allowp, the z-score threshold is not applied.
            This is to avoid rejecting stable data.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with:
            - run_artifact: Boolean indicating if artifact rejection was performed
            - pct_artifact: Percentage of samples identified as artifacts
        - Creates a new column with suffix appended to the current pupil column name
        - Updates all_pupil_cols and all_steps to track processing history
        - Artifact periods are replaced with NaN values
        - Trials with all missing pupil data are skipped and reported
        - Processing parameters are stored in self.params['artifact_rejection']
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
        for group, groupdata in tqdm(grouped, desc=f'Artifact rejection', disable=not self.progress_bar):

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

        # print summary
        print(f"✓ Artifact rejection completed!")
        print(f"  → New column: '{new_col}' (artifacts removed)")
        print(f"  → Previous column '{pupil_col}' preserved.")
        print(f"  → {len(empty_trials)} trial(s) failed.")

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self

    def filter_position(self, vertices, suffix = '_xy'):
        """
        Filter pupil data based on gaze position within a polygon.

        This method removes pupil data points where the gaze position falls outside a specified polygon.
        This is useful for excluding data where participants were not looking at the intended region
        of interest.

        Parameters
        ----------
        vertices : list of tuples
            List of (x,y) coordinates defining the polygon vertices.
            Must be in screen coordinates and form a closed polygon.
            Example: [(0,0), (0,1080), (1920,1080), (1920,0), (0,0)]
        suffix : str, default='_xy'
            Suffix to append to the pupil column name for the filtered data.
            For example, if pupil column is 'pupil', the new column will be 'pupil_xy'.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - Updates summary_data with:
            - run_gaze_filter: Boolean indicating if gaze filtering was performed
            - pct_gaze_filter: Percentage of samples outside the polygon
            - avg_gaze_x: Average gaze x-coordinate for the remaining samples
            - avg_gaze_y: Average gaze y-coordinate for the remaining samples
        - Creates a new column with suffix appended to the current pupil column name
        - Updates all_pupil_cols and all_steps to track processing history
        - Samples outside the polygon are replaced with NaN values
        - Trials with all missing pupil data are skipped and reported
        - Processing parameters are stored in self.params['filter_position']

        Raises
        ------
        ValueError
            If vertices cannot be converted to float numpy array
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
        self.summary_data['run_gaze_filter'] = False
        self.summary_data['pct_gaze_filter'] = pd.NA

        # iterate over trials if trial_identifier is provided
        empty_trials = []
        grouped = self.data.groupby(self.trial_identifier, sort=False)
        for group, groupdata in tqdm(grouped, desc=f'Filtering based on gaze position', disable=not self.progress_bar):
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
                groupdata.loc[groupdata.index[~inside_mask], new_col] = pd.NA
                
                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_gaze_filter'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'pct_gaze_filter'] = 1 - (inside_mask.sum()/len(inside_mask))
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'avg_gaze_x'] = np.nanmean(groupdata.loc[groupdata[new_col].notna(), x_col])
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'avg_gaze_y'] = np.nanmean(groupdata.loc[groupdata[new_col].notna(), y_col])
        
        # update latest pupil column
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Gaze Filtered')

        # print summary
        print(f"✓ Gaze spatial filtering completed!")
        print(f"  → New column: '{new_col}' (gaze filtered)")
        print(f"  → Previous column '{pupil_col}' preserved.")
        print(f"  → {len(empty_trials)} trial(s) failed.")

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {len(empty_trials)} trials not filtered due to missing pupil data:")
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self

    def smooth(self, suffix='_sm', method='hann', window=100, **kwargs):
        """
        Smooth pupil data using various smoothing methods.

        This method applies signal smoothing to reduce noise in the pupil data.
        Three smoothing methods are available:

        1. Rolling mean: Simple moving average
        2. Hann window: Weighted moving average using Hann window
        3. Butterworth filter: Low-pass filter with specified cutoff

        Parameters
        ----------
        suffix : str, default='_sm'
            Suffix to append to the pupil column name for the smoothed data.
            For example, if pupil column is 'pupil', the new column will be 'pupil_sm'.
        method : {'rollingmean', 'hann', 'butter'}, default='hann'
            Method to use for smoothing:
            - 'rollingmean': Simple moving average
            - 'hann': Hann window smoothing
            - 'butter': Butterworth low-pass filter
        window : int, default=100
            Window size (in number of samples) for rolling mean or Hann window smoothing.
            Not used for Butterworth filter.
        **kwargs : dict
            Additional arguments for specific smoothing methods.

            - For rolling mean and hann window:
                Check `pandas.DataFrame.rolling <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html>`_ documentation for additional arguments.
            - For Butterworth filter:
                cutoff_freq : float
                    Cutoff frequency in Hz. Default is 4 Hz.
                order : int
                    Filter order. Default is 3.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining

        Notes
        -----
        - Updates summary_data with smoothing method and parameters
        - Creates a new column with suffix appended to the current pupil column name
        - Updates all_pupil_cols and all_steps to track processing history
        - Missing values (NaN) are preserved
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
        for group, groupdata in tqdm(grouped, desc=f'Smoothing', disable=not self.progress_bar):

            # check if groupdata has any pupil data
            if np.all(groupdata[pupil_col].isna()):
                empty_trials.append(group)
            else:
                if method == 'rollingmean':
                    smoothed = groupdata[pupil_col].rolling(window=window, center=True, **kwargs).mean()
                elif method == 'hann':
                    smoothed = groupdata[pupil_col].rolling(window=window, win_type='hann', center=True, **kwargs).mean()
                elif method == 'butter':
                    smoothed = lowpass_filter(groupdata[pupil_col], sampling_freq=self.sampling_freq, **kwargs)
                self.data.loc[groupdata.index, new_col] = smoothed

                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_smooth'] = True

        # convert to Float64 since some values changed to float64
        self.data[new_col] = self.data[new_col].convert_dtypes()

        # update latest pupil column
        self.all_pupil_cols.append(new_col)
        self.all_steps.append('Smoothed')

        # print summary
        print(f"✓ Smoothing completed!")
        print(f"  → New column: '{new_col}' (smoothed)")
        print(f"  → Previous column '{pupil_col}' preserved.")
        print(f"  → {len(empty_trials)} trial(s) failed.")

        # print empty trials
        if len(empty_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {pd.DataFrame(empty_trials, columns=self.trial_identifier)}")

        return self


    def check_missing(self, pupil_col=None, missing_value=pd.NA):
        """
        Check for missing values in pupil data.

        This method calculates the percentage of missing values for each trial and updates
        the summary statistics. Missing values can be either NaN or a specific value.

        Parameters
        ----------
        pupil_col : str, optional
            Column name to check for missing values.
            If None, uses the latest pupil column.
        missing_value : float or pd.NA, default=pd.NA
            Value to consider as missing.
            Can be pd.NA for NaN values or any specific value.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - Updates summary_data with:
            - run_check_missing: Boolean indicating if missing check was performed
            - missing: Percentage of missing values in each trial
        - Updates all_steps to track processing history
        - Trials that cannot be checked are reported
        - Processing parameters are stored in self.params['check_missing']
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
        for group, groupdata in tqdm(grouped, desc=f'Checking missing values', disable=not self.progress_bar):
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

        # print summary
        print(f"✓ Missing values checked!")
        print(f"  → {len(skip_trials)} trial(s) failed.")
        
        # print failed trials
        if len(skip_trials) > 0:
            print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

        return self


    def interpolate(self, suffix='_it', method='linear', missing_threshold=0.6):
        """
        Interpolate missing values in pupil data.

        This method fills missing values in the pupil data using either linear or spline
        interpolation. Trials with too many missing values (above missing_threshold) are
        skipped to avoid unreliable interpolation.

        Parameters
        ----------
        suffix : str, default='_it'
            Suffix to append to the pupil column name for the interpolated data.
            For example, if pupil column is 'pupil', the new column will be 'pupil_it'.
        method : {'linear', 'spline'}, default='linear'
            Method to use for interpolation:
            - 'linear': Linear interpolation between points
            - 'spline': Cubic spline interpolation
        missing_threshold : float, default=0.6
            Maximum proportion of missing values allowed for interpolation.
            Trials with more missing values than this threshold are skipped.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - Updates summary_data with:
            - run_interpolate: Boolean indicating if interpolation was performed
            - pct_interpolate: Percentage of interpolated values in each trial
        - Creates a new column with suffix appended to the current pupil column name
        - Updates all_pupil_cols and all_steps to track processing history
        - Trials with too many missing values are skipped and reported
        - Processing parameters are stored in self.params['interpolate']

        Raises
        ------
        ValueError
            If method is not 'linear' or 'spline'
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
        for group, groupdata in tqdm(grouped, desc=f'Interpolating', disable=not self.progress_bar):

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

        # print summary
        print(f"✓ Interpolation completed!")
        print(f"  → New column: '{new_col}' (interpolated)")
        print(f"  → Previous column '{pupil_col}' preserved.")
        print(f"  → {len(skip_trials)} trial(s) failed.")

        if len(skip_trials) > 0:
            # print a list of trials with high missing values
            print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

        return self

    def downsample(self, target_samp_freq, agg_methods=None):
        """
        Downsample pupil data to a new sampling rate.

        This method downsamples the data by binning into fixed time windows and aggregating
        values within each bin. This is useful for reducing data size or matching sampling
        rates between different recordings.

        Parameters
        ----------
        target_samp_freq : int
            Target sampling frequency in Hz.
        agg_methods : dict, optional
            Dictionary mapping column names to aggregation methods.
            Example: {'pupil': 'mean', 'time': 'first', 'x': 'mean', 'y': 'mean'}
            If None, uses 'first' for all columns.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - Unlike other preprocessing functions, this function will replace the original .data with the downsampled data rather than creating a new column to the original .data. 
        - Trials that cannot be downsampled are reported. 
        - The sampling frequency is checked and updated again after downsampling. 
        - Updates summary_data with:
            - run_downsample: Boolean indicating if downsampling was performed
            - downsampled_bin_size: Size of the downsampled time bin in milliseconds
            - downsampled_samp_freq: Downsampled sampling frequency in Hz
        - Updates all_steps to track processing history
        - Processing parameters are stored in self.params['downsample']
        """
        # store parameters
        self.params['downsample'] = {k: v for k, v in locals().items() if k != 'self'}

        # calculate new time step in milliseconds
        bin_size_ms = 1000/target_samp_freq

        # get data
        data = self.data
        time_col = self.time_col

        # aggregate methods for downsampling
        aggregation_methods = {col: 'first' for col in data.columns}
        if agg_methods is not None:
            aggregation_methods.update(agg_methods)

        # initialize summary data
        self.summary_data['run_downsample'] = False

        # group data
        grouped = self.data.groupby(self.trial_identifier, sort=False)

        # precompute offsets for each group
        offsets = grouped[time_col].transform('min')

        # normalize time and compute bins
        # this ensures that the first sample in each trial is at time 0 and that the same bin size is used for all trials
        normalized_time = self.data[time_col] - offsets
        bins = normalized_time // bin_size_ms

        # iterate over trials and aggregate data
        skip_trials = []
        all_downsampled = []

        for group, groupdata in tqdm(grouped, desc='Downsampling', disable=not self.progress_bar):
            try:
                # group by bins and aggregate data
                groupdata = groupdata.groupby(bins, as_index=False).agg(aggregation_methods)
                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_downsample'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'downsampled_bin_size'] = bin_size_ms
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'downsampled_samp_freq'] = 1000/bin_size_ms
            except:
                skip_trials.append(group)

            # append downsampled data
            all_downsampled.append(groupdata)
        
        if all_downsampled:
            # concatenate downsampled data
            data = pd.concat(all_downsampled, ignore_index=True)
            
            # check sampling frequency, update samp_freq if pass
            check_pass = self.check_sampling_frequency(sampling_rate=target_samp_freq, data=data)
            if check_pass:
                self.samp_freq = target_samp_freq

                # update data
                self.data = data

                # print summary
                print(f"✓ Downsampling completed!")
                print(f"  → New sampling frequency: {target_samp_freq} Hz")
                print(f"  → {len(skip_trials)} trial(s) failed.")

                # print failed trials
                if len(skip_trials) > 0:
                    print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

                # update latest step
                self.all_steps.append('Downsampled')
        else:
            raise ValueError('No trials were successfully downsampled!')

        return self

    def upsample(self, target_samp_freq, fill_pupil=False):
        """
        Upsample pupil data to a higher sampling rate.

        This method upsamples the data by inserting empty rows to meet the required sampling rate. 
        Missing values are foward-filled for non-pupil columns. Pupil columns remain as NaN where no data exists unless fill_pupil=True.
        A new column 'upsampled' is added to track the inserted rows.
        Trials that cannot be upsampled are reported. The sampling frequency is checked and updated again after upsampling. 

        Parameters
        ----------
        target_samp_freq : int
            Target sampling frequency in Hz.
            Must be higher than current sampling frequency.
        fill_pupil : bool, default=False
            Whether to also fill missing values in pupil columns.
            If False, pupil columns remain as NaN where no data exists.
            This is simply a forward-fill. If you want to interpolate missing values, you can do so after upsampling.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - There might be slight discrepancies in the actual sampling rate from the target sampling rate because the time step between samples is rounded to the nearest integer. For example, if you supply a target sampling rate of 1001 Hz, the actual sampling rate will be 1000 Hz (round(1000/1001)= 1 ms time step). In the current implementation, this will result in an error because the actual sampling rate 1000 Hz does not match the target sampling rate 1001 Hz.
        - Upsampled data can be interpolated to fill missing values. You may need to set a lower missing_threshold for interpolation as the upsampling will introduce more missing values.
        - Updates summary_data with:
            - run_upsample: Boolean indicating if upsampling was performed
            - upsampled_bin_size: Size of the upsampled time bin in milliseconds
            - upsampled_samp_freq: Upsampled sampling frequency in Hz
        - Updates all_steps to track processing history
        - Processing parameters are stored in self.params['upsample']
        """

        # Validate target sampling frequency
        if target_samp_freq <= self.samp_freq:
            raise ValueError(f"Target sampling frequency ({target_samp_freq} Hz) must be higher than current frequency ({self.samp_freq} Hz)")

        # store parameters
        self.params['upsample'] = {k: v for k, v in locals().items() if k != 'self'}

        # calculate new time step in milliseconds
        new_time_step_ms = round(1000 / target_samp_freq)
        
        # get data
        data = self.data
        time_col = self.time_col

        # initialize summary data
        self.summary_data['run_upsample'] = False

        # group data
        grouped = self.data.groupby(self.trial_identifier, sort=False)

        # iterate over trials and upsample data
        skip_trials = []
        all_upsampled = []
        
        for group, groupdata in tqdm(grouped, desc='Upsampling', disable=not self.progress_bar):
            try:
                # get original time range
                min_time = int(groupdata[time_col].min())
                max_time = int(groupdata[time_col].max())
                
                # create complete time series from min_time to max_time 
                complete_time_ms = np.arange(min_time, max_time + new_time_step_ms, new_time_step_ms)
                
                # create new dataframe with complete time series
                new_data = pd.DataFrame({time_col: complete_time_ms})

                # add trial identifier columns
                for i, col in enumerate(self.trial_identifier):
                    if isinstance(group, tuple):
                        new_data[col] = group[i]  # Use integer index for tuple
                    else:
                        new_data[col] = group  # Single column case
                
                # merge with original data to get existing values
                # use outer merge to keep all time points
                merged = pd.merge(new_data, groupdata, on=[time_col] + self.trial_identifier, how='left')
                
                # determine which row is upsampled
                merged['upsampled'] = True
                merged.loc[merged[time_col].isin(groupdata[time_col]), 'upsampled'] = False

                # determine which columns to preserve vs fill
                if fill_pupil:
                    preserve_cols = self.all_pupil_cols + [self.x_col, self.y_col]  # pupil + gaze columns
                else:
                    preserve_cols = [self.x_col, self.y_col]  # only gaze columns
                    
                fill_cols = [col for col in merged.columns if col not in preserve_cols and col not in [time_col] + self.trial_identifier]
                
                # fill missing values for non-preserved columns
                merged[fill_cols] = merged[fill_cols].ffill()
                
                # optionally fill pupil columns if requested
                if fill_pupil:
                    pupil_fill_cols = [col for col in self.all_pupil_cols if col in merged.columns]
                    merged[pupil_fill_cols] = merged[pupil_fill_cols].ffill()
                
                # update summary data
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'run_upsample'] = True
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'upsampled_bin_size'] = new_time_step_ms
                self.summary_data.loc[np.all(self.summary_data[self.trial_identifier] == group, axis=1), 'upsampled_samp_freq'] = target_samp_freq
                
                # append upsampled data
                all_upsampled.append(merged)
                
            except Exception as e:
                skip_trials.append(group)
                print(f"Failed to upsample trial {group}: {str(e)}")

        # concatenate upsampled data
        if all_upsampled:
            data = pd.concat(all_upsampled, ignore_index=True)
            
            # check sampling frequency, update samp_freq if pass
            check_pass = self.check_sampling_frequency(sampling_rate=target_samp_freq, data=data)
            if check_pass:
                self.samp_freq = target_samp_freq

                # update data
                self.data = data
                
                # print summary
                print(f"✓ Upsampling completed!")
                print(f"  → New sampling frequency: {target_samp_freq} Hz")
                print(f"  → {len(skip_trials)} trial(s) failed.")

                # print failed trials
                if len(skip_trials) > 0:
                    print(f"\n {pd.DataFrame(skip_trials, columns=self.trial_identifier)}")

                # update latest step
                self.all_steps.append('Upsampled')
        else:
            raise ValueError("No trials were successfully upsampled!")

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
            Suffix to append to the pupil column name for the corrected data.
            For example, if pupil column is 'pupil', the new column will be 'pupil_bc'.
        method : {'subtractive', 'divisive'}, default='subtractive'
            Method to use for baseline correction:
            - 'subtractive': Subtract baseline mean from pupil data
            - 'divisive': Divide pupil data by baseline mean

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - Updates summary_data with:
            - run_baseline_correction: Boolean indicating if baseline correction was performed
            - baseline: Mean baseline value used for correction
        - Adds a new column with suffix appended to the current pupil column name
        - Updates all_pupil_cols and all_steps to track processing history
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
        for group, groupdata in tqdm(grouped, desc=f'Baseline correction', disable=not self.progress_bar):
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

        # print summary
        print(f"✓ Baseline correction completed!")
        print(f"  → New column: '{new_col}' (baseline corrected)")
        print(f"  → Previous column '{pupil_col}' preserved.")
        print(f"  → {len(skip_trials)} trial(s) failed.")

        # print failed trials
        if len(skip_trials) > 0:
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
            is_outlier = (df_summary['baseline'] > upper) | (df_summary['baseline'] < lower)
            # fill na with False
            is_outlier = is_outlier.fillna(False)
            # update summary data
            df_summary['baseline_outlier'] = is_outlier
            df_summary['baseline_upper'] = upper
            df_summary['baseline_lower'] = lower
        
        else:
            # calculate thresholds for each group
            for group, groupdata in tqdm(df_summary.groupby(outlier_by, sort=False), desc='Checking baseline pupil sizes for outliers', disable=not self.progress_bar):
                # calculate group thresholds using pandas
                median_baseline = groupdata['baseline'].median()
                mad = (groupdata['baseline'] - median_baseline).abs().median()
                upper = median_baseline + n_mad_baseline*mad
                lower = median_baseline - n_mad_baseline*mad
                
                # mark outliers for this group
                group_indices = groupdata.index
                is_outlier = (groupdata['baseline'] > upper) | (groupdata['baseline'] < lower)
                # fill na with False
                is_outlier = is_outlier.fillna(False)
                # update summary data
                df_summary.loc[group_indices, 'baseline_outlier'] = is_outlier
                df_summary.loc[group_indices, 'baseline_upper'] = upper
                df_summary.loc[group_indices, 'baseline_lower'] = lower

        # update summary data
        self.summary_data = df_summary

        # update outlier by
        self.baseline_outlier_by = outlier_by

        # update steps
        self.all_steps.append('Baseline Outlier Detection')
        
        # print summary
        print(f"✓ Baseline outlier detection completed!")
        print(f"  → {df_summary['baseline_outlier'].sum()} trial(s) detected as baseline outliers.")

        # print outlier trials
        if df_summary['baseline_outlier'].any():
            print(f"\n {df_summary.query('baseline_outlier==True')[self.trial_identifier]}")

        # plot if requested
        if plot:
            self.plot_baseline(plot_by=outlier_by, return_fig=False, **kwargs)

        return self
        
    def check_trace_outliers(self, time_col=None, pupil_col=None, outlier_by=None, n_mad_trace=4, plot=True, **kwargs):
        """
        Check for outlier trials based on pupil trace values.

        Detects outlier trials by comparing each trial's pupil trace against thresholds calculated from the median absolute deviation (MAD) of all trials.
        Outliers can be calculated globally or within specified groups.

        Parameters
        ----------
        time_col : str, optional
            Column name for x-axis values (time). Defaults to time column.
        pupil_col : str, optional 
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
        - Updates summary_data with:
            - run_trace_outlier: Boolean indicating if trace outlier detection was performed
            - trace_outlier: Boolean indicating if trial is an outlier
            - trace_upper: Upper threshold for outlier detection
            - trace_lower: Lower threshold for outlier detection
        - Outlier detection uses median absolute deviation (MAD) method
        - Can detect outliers globally or within groups specified by outlier_by
        """

        # get data
        df = self.data.copy()
        df_summary = self.summary_data.copy()

        # get x and y columns
        if time_col is None:
            time_col = self.time_col
        if pupil_col is None:
            pupil_col = self.all_pupil_cols[-1]
        print(f'Checking trace outliers for {pupil_col}')

        # initialize outlier columns
        df_summary['trace_outlier'] = False
        df_summary['trace_upper'] = pd.NA
        df_summary['trace_lower'] = pd.NA

        # calculate outlier thresholds
        if outlier_by is None:
            # calculate thresholds for all trials
            grand_mean = df[pupil_col].mean()
            pupil_dist = df.groupby(self.trial_identifier)[pupil_col].apply(lambda x: (x - grand_mean).abs().max())
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
                grand_mean = groupdata[pupil_col].mean()
                pupil_dist = groupdata.groupby(self.trial_identifier)[pupil_col].apply(lambda x: (x - grand_mean).abs().max())
                median_dist = pupil_dist.median()
                mad = (pupil_dist - median_dist).abs().median()
                
                upper = grand_mean + median_dist + (n_mad_trace * mad)
                lower = grand_mean - median_dist - (n_mad_trace * mad)

                # update summary data for this group
                df_summary.loc[np.all(df_summary[outlier_by] == group, axis=1), 'trace_upper'] = upper
                df_summary.loc[np.all(df_summary[outlier_by] == group, axis=1), 'trace_lower'] = lower

        # mark outliers
        for trial, trialdata in tqdm(df.groupby(self.trial_identifier, sort=False), desc='Checking pupil traces for outliers', disable=not self.progress_bar):
            # get max and min values
            max_val = trialdata[pupil_col].max()
            min_val = trialdata[pupil_col].min()

            # get thresholds for this trial
            trial_mask = np.all(df_summary[self.trial_identifier] == trial, axis=1)
            upper_threshold = df_summary.loc[trial_mask, 'trace_upper'].iloc[0]
            lower_threshold = df_summary.loc[trial_mask, 'trace_lower'].iloc[0]

            # check for outliers and update summary data
            if pd.notna(max_val) and pd.notna(min_val) and pd.notna(upper_threshold) and pd.notna(lower_threshold):
                is_outlier = (max_val > upper_threshold) or (min_val < lower_threshold) 
                df_summary.loc[trial_mask, 'trace_outlier'] = is_outlier

        # update summary data and steps
        self.summary_data = df_summary
        self.all_steps.append('Trace Outlier Detection')

        # update outlier by
        self.trace_outlier_by = outlier_by

        # print summary
        print(f"✓ Trace outlier detection completed!")
        print(f"  → {df_summary['trace_outlier'].sum()} trial(s) detected as trace outliers.")

        # print outlier trials
        if df_summary['trace_outlier'].any():
            print(f"\n {df_summary.query('trace_outlier==True')[self.trial_identifier]}")

        # plot if requested
        if plot:
            self.plot_spaghetti(time_col=time_col, pupil_col=pupil_col, plot_by=outlier_by, return_fig=False, **kwargs)

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
        # convert dtypes
        self.summary_data = self.summary_data.convert_dtypes()

        if columns is None:
            columns = self.summary_data.columns
        if level is None:
            return self.summary_data[columns]
        else:
            if agg_methods is None:
                # get all numeric columns
                numeric_cols = self.summary_data[columns].select_dtypes(include=['number','boolean']).columns
                agg_methods = {col: 'mean' for col in numeric_cols}
                print(f"Using default aggregation methods: {agg_methods}")
            return self.summary_data.groupby(level)[columns].agg(agg_methods)


    def validate_trials(self, trials_to_exclude, invert_mask=False):
        """
        Mark trials as valid/invalid based on exclusion criteria.

        This method adds a 'valid' column to both the data and summary_data, marking
        trials as valid or invalid based on the provided exclusion criteria.

        Parameters
        ----------
        trials_to_exclude : pandas.DataFrame
            DataFrame containing trial identifiers to exclude.
            Must have columns matching the trial_identifier of the PupilProcessor.
        invert_mask : bool, default=False
            If True, excludes all trials except those specified in trials_to_exclude.
            If False, excludes only the trials specified in trials_to_exclude.

        Returns
        -------
        self : PupilProcessor
            Returns self for method chaining.

        Notes
        -----
        - Adds 'valid' column to both summary_data and data
        - Valid column is boolean: True for valid trials, False for invalid trials
        - Trials are matched based on trial_identifier columns
        - Duplicate entries in trials_to_exclude are automatically removed
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

    def plot_pupil_surface(self, data=None, pupil_col=None, x_col=None, y_col=None, plot_type='count', vertices=None, nbins=64, log_counts=False, plot_by=None, show_centroid=True, save=None, plot_params=None):
        """
        Create an interactive surface plot of pupil dilation by gaze coordinates using numpy.histogram2d.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame containing pupil size, x-coordinates, and y-coordinates.
            Being able to specify data is useful for plotting a subset of the data. See examples below.
            If None, uses self.data.
        pupil_col : str, optional
            Column name for pupil size. Defaults to self.all_pupil_cols[-1].
        x_col : str, optional
            Column name for x-coordinates of gaze. Defaults to self.x_col.
        y_col : str, optional
            Column name for y-coordinates of gaze. Defaults to self.y_col.
        plot_type : str, optional
            'count' for number of measurements or 'size' for mean pupil size. Defaults to 'count'.
        nbins : int, optional
            Number of bins for the 2D histogram. Defaults to 64.
        log_counts : bool, default=False
            Whether to apply log transformation to counts (only applies when plot_type='count'). Defaults to False.
        plot_by : str, optional
            Column name to group data by for separate subplots. Defaults to None.
        show_centroid : bool, default=True
            Whether to show the centroid of the data. Defaults to True.
        save : str, optional
            Path to save plot.
        plot_params : dict, optional
            Dictionary of plotting parameters to override defaults
            - x_title : str, default='Gaze X'
            - y_title : str, default='Gaze Y'
            - title : str, default='Pupil Foreshortening Error Surface'
            - palette : str, default='Viridis'
            - width : int, default=400
            - height : int, default=300

        Examples
        --------
        >>> # Plot a 2d histogram of the number of pupil measurements by condition
        >>> p.plot_pupil_surface(plot_by='condition')
        >>> # Plot a 2d histogram of the mean pupil size based on custom data
        >>> p.plot_pupil_surface(data=p.data[p.data['event'] == 'event_name'])
        >>> # Plot the mean pupil size rather than the count of measurements as a function of gaze coordinates 
        >>> p.plot_pupil_surface(plot_type='size')
        """
        # plot params
        if plot_params is None:
            plot_params = {}

        # get data
        if data is None:
            data = self.data.copy()

        # get column names
        if x_col is None:
            x_col = self.x_col
        if y_col is None:
            y_col = self.y_col
        if pupil_col is None:
            pupil_col = self.all_pupil_cols[-1]

        # drop nans
        data = data[data[x_col].notna() & data[y_col].notna() & data[pupil_col].notna()].reset_index(drop=True)

        # get x and y
        x = data[x_col]
        y = data[y_col]
        
        # Create groupby object
        if plot_by is not None:
            # Convert plot_by to list if it's not already
            if isinstance(plot_by, str):
                plot_by = [plot_by]
            grouped = data.groupby(plot_by, sort=False)
        else:
            grouped = [(None, data)]

        # Create figure
        fig = go.Figure()

        # Initialize lists to store all data for common color scale
        all_counts = []
        all_pupil_means = []
        
        # Process each group
        for _, group_data in grouped:
            group_x = group_data[x_col]
            group_y = group_data[y_col]
            group_pupil = group_data[pupil_col]
            
            # Compute histograms for this group
            counts, xedges, yedges = np.histogram2d(group_x, group_y, bins=nbins, range=[[x.min(), x.max()], [y.min(), y.max()]])
            pupil_sum, _, _ = np.histogram2d(group_x, group_y, bins=[xedges, yedges], weights=group_pupil)
            pupil_mean = pupil_sum / np.maximum(counts, 1)

            all_counts.append(counts)
            all_pupil_means.append(pupil_mean)

        # Calculate global min/max for common color scale
        if plot_type == 'count':
            global_min = np.min([np.min(c) for c in all_counts])
            global_max = np.max([np.max(c) for c in all_counts])
            if log_counts:
                global_min = np.log1p(global_min)
                global_max = np.log1p(global_max)
        else:
            global_min = np.min([np.min(pm) for pm in all_pupil_means])
            global_max = np.max([np.max(pm) for pm in all_pupil_means])

        # Create a list to store visibility settings for each trace
        all_traces = []
        visible_settings = []
        dropdown_options = []

        # Plot each group
        for i, (group_name, group_data) in enumerate(grouped):
            traces_in_group = []
            
            # Format dropdown label
            if group_name is not None:
                # Convert group_name to list if it's not already
                if not isinstance(group_name, tuple):
                    group_name = (group_name,)
                # Format each value with leading zeros if numeric
                label_parts = [f"{val:03d}" if isinstance(val, (int, float)) else str(val) for val in group_name]
                label = " | ".join(label_parts)
            else:
                label = "All"

            # Add dropdown option
            dropdown_options.append({
                'label': label,
                'method': "update",
                'args': [{"visible": []}, {}]  # Will be filled later
            })

            # get colorscale data
            if plot_type == 'count':
                colorscale_data = all_counts[i].T
                if log_counts:
                    colorscale_data = np.log1p(colorscale_data)
                colorbar_title = 'Log Count' if log_counts else 'Count'
            else:
                colorscale_data = all_pupil_means[i].T
                colorbar_title = "Mean Size"

            # Add heatmap trace
            heatmap = go.Heatmap(
                x=(xedges[:-1] + xedges[1:]) / 2,
                y=(yedges[:-1] + yedges[1:]) / 2,
                z=colorscale_data,
                colorbar=dict(title=colorbar_title),
                colorscale=plot_params.get('palette', 'Viridis'),
                hoverongaps=False,
                hoverinfo='x+y+z',
                visible=(i == 0),  # Only first group visible initially
                zmin=global_min,
                zmax=global_max
            )
            fig.add_trace(heatmap)
            traces_in_group.append(heatmap)
            
            # Add vertices if provided
            if vertices is not None:
                vertices = np.array(vertices)
                aoi = go.Scatter(
                    x=vertices[:,0],
                    y=vertices[:,1],
                    line=dict(color='black', width=2),
                    name='Custom region',
                    mode='lines',
                    showlegend=False,
                    visible=(i == 0)  # Only first group visible initially
                )
                fig.add_trace(aoi)
                traces_in_group.append(aoi)

            # Add centroid
            if show_centroid:
                centroid = np.array([group_data[x_col].mean(), group_data[y_col].mean()])
                centroid_trace = go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    showlegend=False,
                    hoverinfo='x+y+name',
                    name='Average Gaze',
                    visible=(i == 0)  # Only first group visible initially
                )
                fig.add_trace(centroid_trace)
                traces_in_group.append(centroid_trace)

            all_traces.append(traces_in_group)

        # Create visibility settings for each dropdown option
        for i in range(len(all_traces)):
            vis = []
            for j, traces in enumerate(all_traces):
                vis.extend([True if j == i else False] * len(traces))
            visible_settings.append(vis)
            
            # Update the args for each dropdown option
            dropdown_options[i]['args'][0]["visible"] = vis

        # Update layout
        fig.update_layout(
            title=dict(
                text=plot_params.get('title', 'Pupil Foreshortening Error Surface'),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family='Arial', weight='bold')
            ),
            xaxis_title='Gaze X',
            yaxis_title='Gaze Y',
            yaxis_range=[y.max(), y.min()-1], # invert y axis
            xaxis_range=[x.min()-1, x.max()],
            width=plot_params.get('width', 800),
            height=plot_params.get('height', 600),
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                x=1.0,  # Position the dropdown at the right
                y=1.1,  # Position slightly above the plot
                showactive=True,
                active=0,  # Show first group by default
                buttons=dropdown_options
            )],
            margin=dict(l=80, r=80, t=100, b=80)
        )

        # save figure if path is provided
        if save:
            if save.endswith('.html'):
                fig.write_html(save)
            else:
                raise ValueError(f"Interactive plots must be saved as html file. Got {save}.")

        return fig

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
        kwargs.update({k:v for k,v in plot_params.items() if k not in plot_specific_settings})
        
        return plot_specific_settings, kwargs

    def plot_trial(self, trial, time_col=None, pupil_col=None, hue=None, save=None, interactive=True, plot_params=None):
        """
        Plot data for a single trial.

        A wrapper function that calls either _plot_trial_interactive() or _plot_trial_static() 
        depending on the interactive parameter.

        Parameters
        ----------
        trial : pandas.DataFrame
            DataFrame containing trial identifier.
        time_col : str, optional
            Column name for x-axis values. Defaults to time column specified during initialization.
        pupil_col : str or list, optional
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
            return self._plot_trial_interactive(trial, time_col, pupil_col, hue, save, plot_params)
        else:
            return self._plot_trial_static(trial, time_col, pupil_col, hue, save, plot_params)

    def _plot_trial_static(self, trial, time_col=None, pupil_col=None, hue=None, save=None, plot_params=None):
        """
        Create static plot of trial data using matplotlib.

        Parameters
        ----------
        trial : pandas.DataFrame
            DataFrame containing trial identifier.
        time_col : str, optional
            Column name for x-axis values. Defaults to time column specified during initialization.
        pupil_col : str or list, optional
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
        if time_col is None:
            time_col = self.time_col # default to time column
        if pupil_col is None:
            pupil_col = self.all_pupil_cols # default to all pupil columns

        if isinstance(pupil_col, str):
            pupil_col = [pupil_col] # make sure pupil_col is a list

        # get plot settings
        plot_specific_settings, mpl_kwargs = self._get_plot_settings(time_col, pupil_col, plot_params, is_interactive=False)

        # create subplots with context manager
        with mpl.rc_context(mpl_kwargs):
            fig = plt.figure()
            for i, col in enumerate(pupil_col):
                ax = fig.add_subplot(plot_specific_settings['layout'][0], plot_specific_settings['layout'][1], i+1)
                if hue:
                    for trial_group, groupdata in data.groupby(hue, sort=False):

                        # create label for legend
                        label = ', '.join([str(k) for k in trial_group]) if isinstance(trial_group, tuple) else str(trial_group)
                        ax.plot(groupdata[time_col], groupdata[col], label=label)
                else: # if no hue, plot all data together
                    ax.plot(data[time_col], data[col])

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

    def _plot_trial_interactive(self, trial, time_col=None, pupil_col=None, hue=None, save=None, plot_params=None):
        """
        Create an interactive plot of trial data using Plotly.

        Parameters
        ----------
        trial : pandas.DataFrame
            DataFrame containing trial identifier.
        time_col : str, optional
            Column name for x-axis values. Defaults to time column specified during initialization.
        pupil_col : str or list, optional
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
        if time_col is None:
            time_col = self.time_col # default to time column
        if pupil_col is None:
            pupil_col = self.all_pupil_cols # default to all pupil columns

        if isinstance(pupil_col, str):
            pupil_col = [pupil_col] # make sure pupil_col is a list

        # get plot settings
        plot_specific_settings, ply_kwargs = self._get_plot_settings(time_col, pupil_col, plot_params, is_interactive=True)

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
        for i, col in enumerate(pupil_col):
            # figure out row and column
            curr_row = int(i // plot_specific_settings['layout'][1] + 1)
            curr_col = int(i % plot_specific_settings['layout'][1] + 1)
            if hue:
                for g, (trial_group, groupdata) in enumerate(data.groupby(hue, sort=False)):

                    # create label for legend
                    label = ', '.join([str(k) for k in trial_group]) if isinstance(trial_group, tuple) else str(trial_group)
                    # assign color but cycle through colors if more trials than colors
                    curr_color = cols[g % len(cols)]
                    fig.add_trace(go.Scatter(x=groupdata[time_col], y=groupdata[col], 
                                             mode='lines',
                                             name=label,
                                             line=dict(color=curr_color), 
                                             showlegend=ply_kwargs['showlegend'] if i==0 else False # only show legend for first plot
                                             ), 
                                             row=curr_row, col=curr_col)
            else: 
                curr_color = cols[0]
                fig.add_trace(go.Scatter(x=data[time_col], y=data[col], 
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
        mpl_kwargs.update({k:v for k,v in plot_params.items() if k not in plot_specific_settings})

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
        Create interactive histogram plot of baseline pupil sizes using Plotly Express.

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
            warnings.warn(f"Outlier detection was performed by {self.baseline_outlier_by}. Plotting by {plot_by}. The plotted thresholds may be incorrect.")

        # Plot settings
        plot_specific_settings = {
            'title': 'Baseline Pupil Sizes',
            'x_title': 'Baseline Pupil Sizes',
            'y_title': 'Count',
            'vline_color': 'red',
            'vline_style': 'dash',
            'bins': 30
        }

        # Update plot settings if provided
        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings})

        # Get groups
        if plot_by is not None:
            grouped = df_summary.groupby(plot_by, sort=False)
        else:
            grouped = [(None, df_summary)]

        # Create figure
        fig = go.Figure()

        # Create dropdown menu options
        dropdown_options = []
        
        # Keep track of trace indices for each group
        group_traces = []
        
        # Create a temporary matplotlib figure for seaborn to plot into
        temp_fig, temp_ax = plt.subplots()
        
        # Add traces for each group
        for groupid, (group_name, group_data) in enumerate(grouped):
            # Format group name for display
            group_title = ' | '.join([str(x) for x in group_name]) if isinstance(group_name, tuple) else str(group_name) if group_name is not None else "All"
            
            # Keep track of traces for this group
            current_group_traces = []
            
            # Clear the temporary axis
            temp_ax.clear()
            
            if show_outliers:
                # Use seaborn to compute histogram
                sns_hist = sns.histplot(
                    data=group_data,
                    x='baseline',
                    hue='baseline_outlier',
                    bins=plot_specific_settings['bins'],
                    ax=temp_ax,
                    legend=True
                )
                
                # Get all patches and their colors
                all_patches = sns_hist.patches
                n_patches = len(all_patches)
                patches_per_category = n_patches // 2  # Since we have two categories
                
                # Check if there are any outliers in the data
                has_outliers = group_data['baseline_outlier'].any()

                if has_outliers:
                    # Process non-outliers (first half of patches) and outliers (second half)
                    categories = [
                        (False, '#DD8452', all_patches[:patches_per_category], 'Outliers'),
                        (True, '#4C72B0', all_patches[patches_per_category:], 'Non-outliers')
                    ]
                else:
                    # If no outliers, use all patches with a single color
                    categories = [
                        (False, '#4C72B0', all_patches, 'Non-outliers')
                    ]
                
                for outlier_status, color, patches, label in categories:
                    if patches:  # Only add trace if there are bars
                        # Extract x and y values from patches
                        x = [p.get_x() + p.get_width()/2 for p in patches]
                        y = [p.get_height() for p in patches]
                        widths = [p.get_width() for p in patches]
                        
                        # Add trace to plotly figure
                        fig.add_trace(
                            go.Bar(
                                x=x,
                                y=y,
                                width=widths[0],  # All widths should be the same
                                name=label,
                                marker=dict(line=dict(color='black', width=1.5)),
                                marker_color=color,
                                opacity=0.75,
                                visible=(groupid == 0),
                                hovertemplate="Baseline: %{x}<br>Count: %{y}<extra></extra>"
                            )
                        )
                        current_group_traces.append(len(fig.data) - 1)
                
                # Get thresholds
                upper_thresh = group_data['baseline_upper'].values[0]
                lower_thresh = group_data['baseline_lower'].values[0]
                
                # Get max y value from the histogram
                max_y = max(p.get_height() for p in all_patches) * 1.1
                
                # Add threshold lines
                fig.add_trace(
                    go.Scatter(
                        x=[upper_thresh, upper_thresh],
                        y=[0, max_y],
                        mode='lines',
                        name=f'Upper threshold: {upper_thresh:.2f}',
                        line=dict(
                            color=plot_specific_settings['vline_color'],
                            dash=plot_specific_settings['vline_style'],
                            width=2
                        ),
                        visible=(groupid == 0),
                        showlegend=True
                    )
                )
                current_group_traces.append(len(fig.data) - 1)
                
                fig.add_trace(
                    go.Scatter(
                        x=[lower_thresh, lower_thresh],
                        y=[0, max_y],
                        mode='lines',
                        name=f'Lower threshold: {lower_thresh:.2f}',
                        line=dict(
                            color=plot_specific_settings['vline_color'],
                            dash=plot_specific_settings['vline_style'],
                            width=2
                        ),
                        visible=(groupid == 0),
                        showlegend=True
                    )
                )
                current_group_traces.append(len(fig.data) - 1)
            
            else:
                # Use seaborn to compute histogram without outlier distinction
                sns_hist = sns.histplot(
                    data=group_data,
                    x='baseline',
                    bins=plot_specific_settings['bins'],
                    ax=temp_ax
                )
                
                # Extract histogram data
                patches = sns_hist.patches
                x = [p.get_x() + p.get_width()/2 for p in patches]
                y = [p.get_height() for p in patches]
                widths = [p.get_width() for p in patches]
                
                # Add trace to plotly figure
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        width=widths[0],  # All widths should be the same
                        name='All trials',
                        marker_color='#4C72B0',
                        marker=dict(line=dict(color='black', width=1.5)),
                        opacity=0.75,
                        visible=(groupid == 0),
                        showlegend=False,
                        hovertemplate="Baseline: %{x}<br>Count: %{y}<extra></extra>"
                    )
                )
                current_group_traces.append(len(fig.data) - 1)
            
            # Store traces for this group
            group_traces.append({
                'title': group_title,
                'traces': current_group_traces
            })
            
        # Clean up temporary matplotlib figure
        plt.close(temp_fig)

        # Create dropdown menu options
        for group_info in group_traces:
            # Create visibility settings
            vis = [False] * len(fig.data)
            for trace_idx in group_info['traces']:
                vis[trace_idx] = True
            
            # Add dropdown option with proper title update
            dropdown_options.append(
                dict(
                    args=[
                        {"visible": vis},
                        {
                            "title": {
                                "text": f"{plot_specific_settings['title']} - {group_info['title']}",
                                "x": 0.5,
                                "xanchor": "center",
                                "y": 0.95,
                                "yanchor": "top",
                                "font": {
                                    "size": default_plotly['title_font_size'],
                                    "family": default_plotly['title_font_family'],
                                    "weight": default_plotly['title_font_weight']
                                }
                            }
                        }
                    ],
                    label=group_info['title'],
                    method="update"
                )
            )

        # Update layout with Plotly defaults and dropdown menu
        ply_kwargs = default_plotly.copy()
        ply_kwargs['width'] = plot_params.get('width', 800)
        ply_kwargs['height'] = plot_params.get('height', 500)
        
        # Set initial title to include the first group's name for consistency
        initial_group_title = group_traces[0]['title'] if group_traces else "All"
        ply_kwargs['title'] = {
            "text": f"{plot_specific_settings['title']} - {initial_group_title}",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "top",
            "font": {
                "size": default_plotly['title_font_size'],
                "family": default_plotly['title_font_family'],
                "weight": default_plotly['title_font_weight']
            }
        }
        
        ply_kwargs['xaxis_title'] = plot_specific_settings['x_title']
        ply_kwargs['yaxis_title'] = plot_specific_settings['y_title']
        ply_kwargs['updatemenus'] = [{
            'buttons': dropdown_options,
            'direction': 'down',
            'showactive': True,
            'x': 1.2,
            'y': 1.2,
            'xanchor': 'right',
            'yanchor': 'top'
        }]
        ply_kwargs['yaxis_range'] = [0, max_y]
        ply_kwargs['barmode'] = 'overlay'
        ply_kwargs.update({k:v for k,v in plot_params.items() if k not in plot_specific_settings})
        
        # update layout
        fig.update_layout(**ply_kwargs)

        # Save if requested
        if save:
            if save.endswith('.html'):
                fig.write_html(save)
            else:
                raise ValueError(f"Interactive plots must be saved as html file. Got {save}.")

        # Return or display figure
        if return_fig:
            return fig
        else:
            display(fig)

    def plot_spaghetti(self, time_col=None, pupil_col=None, show_outliers=True, plot_by=None, save=False, plot_params=None, return_fig=True): 
        """
        Plot pupil traces for all trials as a spaghetti plot.

        Parameters
        ----------
        time_col : str, optional
            Column name for x-axis. Defaults to time column specified during initialization.
        pupil_col : str, optional 
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
        if time_col is None:
            time_col = self.time_col # default to time column
        if pupil_col is None:
            pupil_col = self.all_pupil_cols[-1] # default to last pupil column

        # get data
        df_plot = self.data.copy()
        if plot_by is not None:
            # convert plot_by to list if not already
            if isinstance(plot_by, str):
                plot_by = [plot_by]
            # get unique columns
            cols = [time_col, pupil_col] + plot_by + self.trial_identifier 
            cols = list(set(cols))
            grouped = df_plot[cols].groupby(plot_by, sort=False)
        else:
            cols = [time_col, pupil_col] + self.trial_identifier
            cols = list(set(cols))
            grouped = [(None, df_plot[cols])]
        
        # Get overall x range for threshold lines
        x_min = df_plot[time_col].min()
        x_max = df_plot[time_col].max()
        
        # check if outlier by is the same as plot_by
        if show_outliers and self.trace_outlier_by is not None and self.trace_outlier_by != plot_by: 
            # both outlier by and plot by should be a list at this point
            warnings.warn(f"Outlier detection was performed by {self.trace_outlier_by}. Plotting by {plot_by}. The plotted thresholds may be incorrect.")

        # plot
        # some additional plot settings specific to this plot
        plot_specific_settings = {
            'title': 'Spaghetti plot',
            'subplot_titles': [' | '.join([str(x) for x in group]) for group, _ in grouped] if plot_by is not None else None,
            'x_title': time_col,
            'y_title': pupil_col,
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
        ply_kwargs.update({k:v for k,v in plot_params.items() if k not in plot_specific_settings})

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
                if self.samp_freq > 100:
                    downsample_mask = np.arange(len(trialdata)) % 10 == 0
                    downsampled = trialdata[downsample_mask]
                else:
                    downsampled = trialdata

                trace = go.Scatter(
                    x=downsampled[time_col],
                    y=downsampled[pupil_col],
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
                    x=[x_min, x_max],  # Use overall x range
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
                    x=[x_min, x_max],  # Use overall x range
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
            dropdown_options[i]['args'][1] = {}  # Empty dict to avoid title updates
        
        # Update layout to include dropdown menu and set fixed title
        fig.update_layout(
            title=plot_specific_settings['title'],  # Set fixed title
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
            if save.endswith('.html'):
                fig.write_html(save)
            else:
                raise ValueError(f"Interactive plots must be saved as html file. Got {save}.")

        # return figure if requested
        if return_fig:
            return fig
        else:
            display(fig)


    def plot_evoked(self, data=None, pupil_col=None, condition=None, agg_by=None, error='ci', save=None, plot_params=None, **kwargs):
        """
        Plot evoked pupil response.

        Creates plot of average pupil response across trials, optionally split by condition and aggregated by specified groups.

        Parameters
        ----------
        data : str or pandas.DataFrame, optional
            Data to plot. If string, uses corresponding attribute.
        pupil_col : str, optional
            Column name for pupil values.
        condition : str or list, optional
            Column(s) to split data by.
        agg_by : str or list, optional
            Column(s) to aggregate data by before computing mean trace and confidence bands.
            For example, to compute subject-level means, use 'subject_id'.
        error : {'ci', 'sem', 'std', None}, default='ci'
            Type of error to plot:
            - 'ci': bootstrap confidence interval
            - 'sem': standard error of the mean
            - 'std': standard deviation
            - None: no error bars
        save : str, optional
            Path to save plot.
        plot_params : dict, default={}
            Additional plotting parameters. This includes all rcParams accepted by matplotlib, as well as the following:
            - 'title': title of plot
            - 'x_title': x-axis label
            - 'y_title': y-axis label
            - 'vline_color': color of vertical line
            - 'vline_linestyle': linestyle of vertical line
            - 'grid': whether to show grid
            - 'legend_labels': labels for legend
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

        # handle agg_by
        if agg_by is not None:
            if isinstance(agg_by, str):
                agg_by = [agg_by]

        # get minimum length across all trials
        min_len = data.groupby(self.trial_identifier, sort=False)[pupil_col].count().min()
        print(f'Data will be padded to minimum length: {min_len} samples')

        # if no condition, process all data together
        if condition is None:
            if agg_by is not None:
                # First compute mean trace for each aggregation group
                agg_traces = []
                for group, group_data in data.groupby(agg_by, sort=False):
                    # Get all trials for this group and compute mean
                    trials = group_data.groupby(self.trial_identifier, sort=False)
                    group_array = np.empty((trials.ngroups, min_len))
                    
                    for i, (_, trial_data) in enumerate(trials):
                        vals = np.asarray(trial_data[pupil_col].to_list())
                        vals = vals[:min_len]
                        group_array[i,:] = vals
                    
                    # Store mean trace for this group
                    agg_traces.append(np.nanmean(group_array, axis=0))
                
                # Convert to array for plotting
                test_array = np.array(agg_traces)
                n_groups = len(agg_traces)
                print(f'Computing average from {n_groups} {agg_by} means')
            else:
                # Process all trials without aggregation
                grouped = data.groupby(self.trial_identifier, sort=False)
                test_array = np.empty((grouped.ngroups, min_len))
                
                for i, (_, trial_data) in enumerate(grouped):
                    vals = np.asarray(trial_data[pupil_col].to_list())
                    vals = vals[:min_len]
                    test_array[i,:] = vals
                
                print(f'Computing average from {grouped.ngroups} trials')

            arrays_by_condition = {'all': test_array}
            
        else:
            # Get actual combinations from data
            arrays_by_condition = {}
            
            # Get unique combinations of conditions that exist in the data
            condition_combinations = data[condition].drop_duplicates()
            
            for _, comb in condition_combinations.iterrows():
                # create mask for this combination
                mask = pd.Series(True, index=data.index)
                for cond in condition:
                    mask &= (data[cond] == comb[cond])
                
                # get data for this combination
                subset = data[mask]
                
                if agg_by is not None:
                    # First compute mean trace for each aggregation group
                    agg_traces = []
                    for group, group_data in subset.groupby(agg_by, sort=False):
                        # Get all trials for this group and compute mean
                        trials = group_data.groupby(self.trial_identifier, sort=False)
                        group_array = np.empty((trials.ngroups, min_len))
                        
                        for i, (_, trial_data) in enumerate(trials):
                            vals = np.asarray(trial_data[pupil_col].to_list())
                            vals = vals[:min_len]
                            group_array[i,:] = vals
                        
                        # Store mean trace for this group
                        agg_traces.append(np.nanmean(group_array, axis=0))
                    
                    # Convert to array for plotting
                    test_array = np.array(agg_traces)
                    n_groups = len(agg_traces)
                    print(f'Condition {comb.to_dict()}: Computing average from {n_groups} {agg_by} means')
                else:
                    # Process all trials without aggregation
                    grouped = subset.groupby(self.trial_identifier, sort=False)
                    test_array = np.empty((grouped.ngroups, min_len))
                    
                    for i, (_, trial_data) in enumerate(grouped):
                        vals = np.asarray(trial_data[pupil_col].to_list())
                        vals = vals[:min_len]
                        test_array[i,:] = vals
                    
                    print(f'Condition {comb.to_dict()}: Computing average from {grouped.ngroups} trials')
                
                # store array with condition name
                cond_name = '_'.join([f'{v}' for v in comb.values])
                arrays_by_condition[cond_name] = test_array

        # plot settings
        plot_specific_settings = {
        'title': 'Task Evoked Pupillary Response',
        'x_title': 'Time (s)',
        'y_title': 'Pupil Size Change',
        'vline_color': 'red',
        'vline_linestyle': '--',
        'grid': False,
        'legend_labels': list(arrays_by_condition.keys())
        }

        plot_specific_settings.update({k:v for k,v in plot_params.items() if k in plot_specific_settings})
        mpl_kwargs = default_mpl.copy()
        mpl_kwargs.update({k:v for k,v in plot_params.items() if k not in plot_specific_settings})

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
        Save PupilProcessor object to file using dill serialization.

        This method saves the entire PupilProcessor object, including all data and processing
        history, to a file for later use.

        Parameters
        ----------
        path : str
            Path where the object should be saved.
            Should include the file extension (e.g., '.pkl').

        Raises
        ------
        FileExistsError
            If a file already exists at the specified path.
        """
        # check if file exists
        if os.path.exists(path):
            raise FileExistsError(f"File {path} already exists.")
        
        # save data
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load PupilProcessor object from file using dill deserialization.

        This method loads a previously saved PupilProcessor object, restoring all data
        and processing history.

        Parameters
        ----------
        path : str
            Path to the file containing the saved PupilProcessor object.

        Returns
        -------
        PupilProcessor
            The loaded PupilProcessor object.

        Notes
        -----
        - The loaded object will be an exact copy of the saved object
        - All data, parameters, and processing history are preserved
        - Make sure the file was created using the save() method
        """
        # load data
        with open(path, 'rb') as f:
            return dill.load(f) 

    def copy(self):
        """
        Create a deep copy of the PupilProcessor object.

        This method creates an independent copy of the PupilProcessor object, including
        all data and processing history. Modifications to the copy will not affect the
        original object.

        Returns
        -------
        PupilProcessor
            A deep copy of the current object.

        Notes
        -----
        - Creates a completely independent copy using copy.deepcopy
        - All data, parameters, and processing history are copied
        - Useful for creating alternative processing pipelines
        """
        import copy
        # deepcopy
        return copy.deepcopy(self)

    @staticmethod
    def combine(processors):
        """
        Combine multiple PupilProcessor instances into a single instance.

        This method allows combining data from multiple processors that have gone through
        identical preprocessing pipelines. This is useful for:
        1. Processing large datasets in chunks to manage memory
        2. Adding new data to an existing processed dataset
        3. Processing data from multiple participants separately

        Parameters
        ----------
        processors : list of PupilProcessor
            List of PupilProcessor instances to combine.
            All processors must have identical preprocessing settings.

        Returns
        -------
        PupilProcessor
            A new PupilProcessor instance containing combined data.

        Notes
        -----
        - All processors must have identical:
            - Initialization parameters (pupil_col, time_col, etc.)
            - Data structure (column names and order)
            - Preprocessing steps and parameters
            - Outlier detection settings (if used)
        - Data and summary statistics are concatenated

        Raises
        ------
        ValueError
            If processors have different preprocessing settings
            If processors have incompatible data structures
            If no processors are provided
        """
        if not processors:
            raise ValueError("No processors provided")
        if len(processors) == 1:
            return processors[0].copy()
            
        # Use the first processor as reference
        ref = processors[0]
        
        # Check compatibility of all processors
        for i, proc in enumerate(processors[1:], 1):
            # Check initialization parameters and data structure
            init_attrs = [
                'pupil_col',      # Original pupil column
                'time_col',       # Time column
                'x_col',          # X position column
                'y_col',          # Y position column
                'samp_freq',      # Sampling frequency
                'trial_identifier', # Trial identifier columns
                'recording_unit', # Recording unit
                'artificial_d',   # Artificial pupil diameter
                'artificial_size' # Artificial pupil size
            ]
            
            # Check if all initialization attributes match
            for attr in init_attrs:
                if not hasattr(proc, attr) or getattr(proc, attr) != getattr(ref, attr):
                    raise ValueError(f"Processor {i} has different {attr} than the reference processor")
            
            # Check if data columns match (both names and order)
            if not list(proc.data.columns) == list(ref.data.columns):
                raise ValueError(f"Processor {i} has different columns or column order than the reference processor")
            
            # Check preprocessing steps by comparing pupil column names and steps
            if not proc.all_pupil_cols == ref.all_pupil_cols:
                raise ValueError(f"Processor {i} has different pupil columns than the reference processor")
                
            if not proc.all_steps == ref.all_steps:
                raise ValueError(f"Processor {i} has different preprocessing steps than the reference processor.\nReference steps: {ref.all_steps}\nProcessor {i} steps: {proc.all_steps}")
            
            # Check if preprocessing parameters match for each step
            # Get all unique parameter keys
            all_keys = set(ref.params.keys()) | set(proc.params.keys())
            diff_params = {}
            missing_params = set(ref.params.keys()) - set(proc.params.keys())
            extra_params = set(proc.params.keys()) - set(ref.params.keys())
            
            # Compare parameters that exist in both
            for k in ref.params.keys() & proc.params.keys():
                ref_val = ref.params[k]
                proc_val = proc.params[k]
                
                # Compare dictionaries within parameters
                if isinstance(ref_val, dict) and isinstance(proc_val, dict):
                    ref_dict = {key: str(val) if hasattr(val, 'shape') else val 
                              for key, val in ref_val.items() if key != 'self'}
                    proc_dict = {key: str(val) if hasattr(val, 'shape') else val 
                               for key, val in proc_val.items() if key != 'self'}
                    if ref_dict != proc_dict:
                        diff_params[k] = (ref_dict, proc_dict)
                else:
                    # For non-dictionary values, convert to string if they're array-like
                    ref_str = str(ref_val) if hasattr(ref_val, 'shape') else ref_val
                    proc_str = str(proc_val) if hasattr(proc_val, 'shape') else proc_val
                    if ref_str != proc_str:
                        diff_params[k] = (ref_str, proc_str)
            
            if diff_params or missing_params or extra_params:
                error_msg = f"Processor {i} has different preprocessing parameters than the reference processor.\n"
                if diff_params:
                    error_msg += "Different parameters:\n"
                    for param, (ref_val, proc_val) in diff_params.items():
                        error_msg += f"  {param}: reference={ref_val}, processor{i}={proc_val}\n"
                if missing_params:
                    error_msg += f"Missing parameters: {missing_params}\n"
                if extra_params:
                    error_msg += f"Extra parameters: {extra_params}\n"
                raise ValueError(error_msg)
            
            # Check if outlier detection settings match
            outlier_attrs = [
                'baseline_outlier_by',  # Grouping for baseline outliers
                'trace_outlier_by',     # Grouping for trace outliers
                'baseline_query',       # Baseline selection query
                'baseline_range'        # Baseline time range
            ]
            
            for attr in outlier_attrs:
                if hasattr(ref, attr):  # Only check if reference has this attribute
                    if not hasattr(proc, attr) or getattr(proc, attr) != getattr(ref, attr):
                        raise ValueError(f"Processor {i} has different {attr} than the reference processor")
            
            # Check summary data structure if it exists
            if proc.summary_data is not None and ref.summary_data is not None:
                if not list(proc.summary_data.columns) == list(ref.summary_data.columns):
                    raise ValueError(f"Processor {i} summary data has different columns or column order than the reference processor")
        
        # Create new processor with combined data
        combined = ref.copy()
        
        # Combine data from all processors
        data_frames = [p.data for p in processors]
        combined.data = pd.concat(data_frames, axis=0, ignore_index=True)
        
        # Update trials attribute to reflect combined data
        combined.trials = combined.data[combined.trial_identifier].drop_duplicates().reset_index(drop=True)
        
        # Combine summary data if it exists in all processors
        if all(p.summary_data is not None for p in processors):
            summary_frames = [p.summary_data for p in processors]
            combined.summary_data = pd.concat(summary_frames, axis=0, ignore_index=True)
        
        return combined

def compute_speed(x, y):
    """
    Compute the speed of change between two arrays.

    This function calculates the rate of change (speed) between corresponding points in
    two arrays. The speed is computed as the absolute maximum of the forward and backward
    differences at each point, normalized by the time difference.

    Parameters
    ----------
    x : array-like
        First array of values, typically pupil measurements.
        Must be numeric and same length as y.
    y : array-like 
        Second array of values, typically time points.
        Must be numeric and same length as x.

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

    This function converts pupil measurements from raw units (arbitrary units from the
    eye tracker) to millimeters using calibration values from an artificial pupil.
    It handles both diameter and area measurements.

    Parameters
    ----------
    pupil_size : float or array-like
        Pupil size in recording units (diameter or area).
        Can be a single value or an array of measurements.
    artificial_d : float
        Diameter of artificial pupil used for calibration (in mm).
        This is the known physical size of the calibration pupil.
    artificial_size : float
        Size of artificial pupil in recording units (diameter or area).
        This is the size measured by the eye tracker for the calibration pupil.
    recording_unit : {'diameter', 'area'}, default='diameter'
        Unit of the recorded measurements:
        - 'diameter': Linear scaling is applied
        - 'area': Square root is taken before scaling

    Returns
    -------
    numpy.ndarray
        Converted pupil measurements in millimeters.
        Will have same shape as input pupil_size.

    Notes
    -----
    - The unit of artificial_size must match the recording_unit
    - The unit of artificial_d is always in millimeters
    - For diameter recordings: output = artificial_d * pupil_size / artificial_size
    - For area recordings: output = artificial_d * sqrt(pupil_size / artificial_size)
    - Useful for standardizing pupil measurements across different setups

    Raises
    ------
    ValueError
        If recording_unit is not 'diameter' or 'area'

    """
    if recording_unit == 'diameter':
        return artificial_d * pupil_size / artificial_size
    elif recording_unit == 'area':
        return artificial_d * np.sqrt(pupil_size) / np.sqrt(artificial_size)
    else:
        raise ValueError(f"Invalid recording unit: {recording_unit}")

def prf(t, t_max=500, n=10.1):
    """
    PRF function according to Hoeks and Levelt (1993)
    
    Parameters
    ----------
    t : array-like
        Time points in milliseconds.
    t_max : float, optional
        Location of the peak (default is 500 ms).
    n : float, optional
        Scale parameter (default is 10.1).

    Returns
    -------
    numpy.ndarray
        Normalized PRF values at each time point.
    """
    h = (t**n)*np.exp((-n*t)/t_max)

    # normalize
    h = h / np.max(h)

    return h

def _generate_pupil_data(n_participants=6, n_trials=20, 
                       stim_duration_ms=2000, baseline_duration_ms=500, 
                       sampling_rate=1000, design_type='within-subject',
                       condition_names=['A','B'],
                       condition_effect=0.5, seed=1):
    """Generate fake pupillometry data for experimental designs.

    This function generates fake pupil size data that mimics typical task-evoked 
    pupillary responses. It supports both between-subject and within-subject designs 
    with two conditions. The pupil response is generated by convolving an impulse at 
    stimulus onset with a pupil response function (PRF).

    Parameters
    ----------
    n_participants : int, default=6
        Number of participants to simulate. For between-subject designs, this should 
        be even to ensure balanced groups.
    n_trials : int, default=20
        Number of trials per participant. For within-subject designs, this will be 
        adjusted to the nearest even number to ensure balanced conditions.
    stim_duration_ms : int, default=2000
        Duration of the stimulus period in milliseconds.
    baseline_duration_ms : int, default=500
        Duration of the pre-stimulus baseline period in milliseconds.
    sampling_rate : int, default=1000
        Sampling rate in Hz. Determines the temporal resolution of the data.
    design_type : {'between-subject', 'within-subject'}, default='within-subject'
        Type of experimental design:
        - 'between-subject': Each participant is assigned to one condition
        - 'within-subject': Each participant completes trials in both conditions
    condition_names : list of str, default=['A', 'B']
        Names of the two experimental conditions. First name is control condition,
        second name is experimental condition.
    condition_effect : float, default=0.5
        Size of the experimental effect for the second condition relative to the first.
        For example, 0.5 means condition B has 50% larger responses than condition A.
    seed : int, optional, default=1
        Random seed for reproducibility. Set to None for random behavior.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the simulated pupil data with columns:
        - participant: Participant identifier (e.g., 'P1', 'P2', ...)
        - condition: Experimental condition
        - trial: Trial number
        - event: Trial phase ('fixation' or 'stimulus')
        - trialtime: Time points in milliseconds
        - pp: Pupil size values
        - x: Horizontal gaze position
        - y: Vertical gaze position

    Notes
    -----
    These features are hard-coded:
    - Individual differences in baseline pupil size
    - Trial-to-trial variability in response amplitude
    - Random blinks (30% probability per trial)
    - Measurement noise
    - Gaze position drift
    - 50% probability of no response on each trial
    
    The pupil response is generated using the following steps:
    - Create baseline period with participant-specific mean
    - Generate stimulus response by convolving an impulse with PRF
    - Add various sources of noise and artifacts
    - Combine baseline and stimulus periods

    Examples
    --------
    >>> # Generate data for a within-subject design
    >>> data_within = generate_pupil_data(
    ...     n_participants=4,
    ...     n_trials=10,
    ...     design_type='within-subject',
    ...     condition_names=['low_load', 'high_load'],
    ...     condition_effect=0.3
    ... )

    >>> # Generate data for a between-subject design
    >>> data_between = generate_pupil_data(
    ...     n_participants=6,
    ...     n_trials=10,
    ...     design_type='between-subject',
    ...     condition_names=['control', 'treatment'],
    ...     condition_effect=0.5
    ... )
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    if design_type not in ['between-subject', 'within-subject']:
        raise ValueError("design_type must be either 'between-subject' or 'within-subject'")
    
    # For within-subject design, ensure n_trials is even
    if design_type == 'within-subject' and n_trials % 2 != 0:
        n_trials += 1
        print(f"Adjusted n_trials to {n_trials} for balanced design")
    
    # Assign participants to conditions for between-subject design
    if design_type == 'between-subject':
        participant_conditions = {}
        for p in range(1, n_participants + 1):
            # Ensure balanced assignment to conditions
            condition = condition_names[0] if p <= n_participants // 2 else condition_names[1]
            participant_conditions[f'P{p}'] = condition
    
    all_data = []
    
    # Pre-compute PRF kernel for convolution
    kernel_time = np.arange(stim_duration_ms)
    t_max = 1200
    prf_kernel = prf(kernel_time, t_max=t_max)  # peak at 1500ms
    
    for p in range(1, n_participants + 1):
        participant_id = f'P{p}'
        participant_baseline = np.random.normal(3.0, 0.05)
        
        # Create balanced sequence of conditions for within-subject design
        if design_type == 'within-subject':
            conditions = [condition_names[0]] * (n_trials // 2) + [condition_names[1]] * (n_trials // 2)
            np.random.shuffle(conditions)
        
        for t in range(1, n_trials + 1):
            # Determine condition for this trial
            if design_type == 'between-subject':
                condition = participant_conditions[participant_id]
            else:  # within-subject
                condition = conditions[t-1]
            
            # Calculate number of samples
            n_samples_stim = int(stim_duration_ms * (sampling_rate/1000))
            n_samples_baseline = int(baseline_duration_ms * (sampling_rate/1000))
            
            # Generate baseline period
            trial_baseline = participant_baseline + np.random.normal(0, 0.05)
            baseline_data = trial_baseline * np.ones(n_samples_baseline)
            baseline_data += np.random.normal(0, 0.05, n_samples_baseline)  # Add noise to baseline
            
            # Generate stimulus response
            # Create impulse at stimulus onset (t=0 in stimulus period)
            impulse = np.zeros(n_samples_stim)
            impulse[0] = 1.0
            
            # Add condition effect and trial-to-trial variability
            amplitude = (condition_effect if condition == condition_names[1] else 1.0)
            amplitude *= np.random.normal(1, 0.5)  # Add trial-to-trial variability
            # sometimes there is no response
            if np.random.random() < 0.5:
                amplitude = 0
            impulse[0] *= amplitude
            
            # Convolve with PRF to get stimulus response
            response = np.convolve(impulse, prf_kernel, mode='full')[:n_samples_stim]
            stim_data = trial_baseline + response
            stim_data += np.random.normal(0, 0.05, n_samples_stim)  # Add noise to stimulus period
            
            # Combine baseline and stimulus data
            pupil = np.concatenate([baseline_data, stim_data])
            time = np.arange(len(pupil))
            event = ['fixation']*n_samples_baseline + ['stimulus']*n_samples_stim
            
            # Add random blinks
            if np.random.random() < 0.3:
                blink_start = np.random.randint(0, len(pupil) - 200)
                blink_duration = np.random.randint(100, 200)
                blink_idx = np.arange(blink_start, blink_start + blink_duration)
                blink_idx = blink_idx[blink_idx < len(pupil)]
                pupil[blink_idx] = 0
            
            # Add gaze position with drift
            drift_x = np.cumsum(np.random.normal(0, 0.01, len(pupil)))
            drift_y = np.cumsum(np.random.normal(0, 0.01, len(pupil)))
            x = np.random.normal(1920/2, 20, len(pupil)) + drift_x
            y = np.random.normal(1080/2, 20, len(pupil)) + drift_y
            
            trial_data = pd.DataFrame({
                'participant': participant_id,
                'condition': condition,
                'trial': t,
                'event': event,
                'trialtime': time,
                'pp': pupil,
                'x': x,
                'y': y
            })
            
            all_data.append(trial_data)
    
    samples = pd.concat(all_data, ignore_index=True).convert_dtypes()
    return samples
    