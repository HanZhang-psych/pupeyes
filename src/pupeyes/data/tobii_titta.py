# -*- coding:utf-8 -*-

"""
Tobii Data Parsing Module (from Titta)

This module is designed for parsing Tobii data saved from a Titta experiment (hdf5 format). It provides functionalities to parse messages and raw gaze samples. 
However, it does not support parsing fixations, saccades, and blinks, as these are not saved by Titta.

For more info on the Titta package, see https://github.com/marcus-nystrom/Titta
"""

import pandas as pd
import numpy as np
import h5py

class TobiiTittaReader:
    """
    A class to read and parse Tobii data saved from Titta (hdf5 format).
    This class handles loading and parsing of Tobii data files, providing methods to extract messages and gaze samples.
    However, it does not support parsing fixations, saccades, and blinks, as these are not saved by Titta.

    Most functions here are wrappers for existing functionalities in the Titta package.

    Parameters
    ----------
    path : str
        Path to the Tobii hdf5 data file
    start_msg : str
        Common part of message marking the start of a trial. For example, if your trial start messages are
        'TRIAL_START 1 1', 'TRIAL_START 1 2', etc., then start_msg would be 'TRIAL_START'
    stop_msg : str 
        Common part of message marking the end of a trial. For example, if your trial end messages are
        'TRIAL_END 1 1', 'TRIAL_END 1 2', etc., then stop_msg would be 'TRIAL_END'
    msg_format : dict
        Dictionary specifying the format of messages. The messages will be parsed based on this format.
        Example: {'marker': str, 'event': str, 'block': int, 'trial': int}
    delimiter : str
        Character used to separate message components. For example, if messages are formatted as 'TRIAL_END 1 1',
        the delimiter would be ' '.
    add_cols : dict, optional
        Additional columns to add to output DataFrames. The dictionary should be in the format {'column_name': column_data}. 
        For example, to add a column 'subject' with value 'S01' to all rows, use {'subject': 'S01'}.
    progress_bar : bool, optional
        If True, shows a progress bar while reading the data file. Default is True.

    Attributes
    ----------
    calibration_history : pd.DataFrame
        Raw calibration history as saved by Titta
    external_signal : pd.DataFrame
        Raw external signal as saved by Titta
    gaze : pd.DataFrame
        Raw gaze data as saved by Titta
    log : pd.DataFrame
        Raw log data as saved by Titta
    msg : pd.DataFrame
        Raw message data as saved by Titta
    notification : pd.DataFrame
        Raw notification data as saved by Titta
    time_sync : pd.DataFrame
        Raw time sync data as saved by Titta

    Examples
    --------
    >>> reader = TobiiTittaReader(
    ...     path='subject01.h5',
    ...     start_msg='TRIAL_START',
    ...     stop_msg='TRIAL_END',
    ...     msg_format={'marker': str, 'event': str, 'block': int, 'trial': int},
    ...     delimiter=' '
    ... )
    """

    def __init__(self, path, start_msg, stop_msg, msg_format, delimiter, add_cols=None):
        """
        Initialize TobiiTittaReader for processing eye tracking data.

        Parameters
        ----------
        path : str
            Path to the Tobii hdf5 data file
        start_msg : str
            Common part of message marking the start of a trial. For example: 'TRIAL_START'
        stop_msg : str 
            Common part of message marking the end of a trial. For example: 'TRIAL_END'
        msg_format : dict
            Dictionary specifying the format of messages. The messages will be parsed based on this format.
            Example: {'marker': str, 'event': str, 'block': int, 'trial': int}
        delimiter : str
            Character used to separate message components. For example: ' '
        add_cols : dict, optional
            Additional columns to add to output DataFrames. The dictionary should be in the format {'column_name': column_data}. 
            For example, to add a column 'subject' with value 'S01' to all rows, use {'subject': 'S01'}.
        """
        self.path = path
        self.start_msg = start_msg
        self.stop_msg = stop_msg
        self.msg_format = msg_format
        self.delimiter = delimiter
        self.add_cols = add_cols
        # read data from hdf5 file
        self.calibration_history = None
        self.external_signal = None
        self.gaze = None
        self.log = None
        self.msg = None
        self.notification = None
        self.time_sync = None
        
        # Read data from hdf5 file
        with h5py.File(path, 'r') as h5_file:
            dataset_names = h5_file.keys()
            if 'calibration_history' in dataset_names:
                self.calibration_history = pd.read_hdf(path, 'calibration_history')
            if 'external_signal' in dataset_names:
                self.external_signal = pd.read_hdf(path,'external_signal')
            if 'gaze' in dataset_names:
                self.gaze = pd.read_hdf(path,'gaze')
            if 'log' in dataset_names:
                self.log = pd.read_hdf(path,'log')
            if 'msg' in dataset_names:
                self.msg = pd.read_hdf(path,'msg')
            if 'notification' in dataset_names:
                self.notification = pd.read_hdf(path,'notification')
            if 'time_sync' in dataset_names:
                self.time_sync = pd.read_hdf(path,'time_sync')

    def get_messages(self):
        """
        Extract and process marker events from the Titta dataset.

        This method extracts all message events from the data and parses them according
        to the specified message format and delimiter.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed message data with columns:

            - id : int
                Trial identifier
            - system_time_stamp : float
                System timestamps
            - msg : str
                Raw message string
            - Additional columns to store parsed message parts based on msg_format specification.
            - Additional columns from self.add_cols are added if specified.

        Notes
        -----
        Messages are split using the specified delimiter and parsed according
        to the data types specified in msg_format.
        """
        # filter messages for start and stop messages
        messages = self.msg[(self.msg.msg.str.contains(self.start_msg)) | (self.msg.msg.str.contains(self.stop_msg))].reset_index(drop=True)
        
        # whenever a new start_msg is found, adds 1 to trial id
        messages['id'] = np.cumsum(messages.msg.str.contains(self.start_msg))-1
        
        # reorder columns
        messages = messages[['id', 'system_time_stamp','msg']]
        
        # parse message parts
        for m, col in enumerate(self.msg_format.keys()):
            messages[col] = messages.msg.str.split(pat=self.delimiter, expand=True)[m].astype(self.msg_format[col])
        
        # Add any additional columns
        if self.add_cols:
            messages = messages.assign(**(self.add_cols))
        
        # sort by system_time_stamp
        messages = messages.sort_values('system_time_stamp')

        return messages

    def get_samples(self, parse_messages=True):
        """
        Extract gaze samples for each trial based on start and stop messages.

        Parameters
        ----------
        parse_messages : bool, optional
            If True, parse message columns and add them to samples. If False, only add raw message.
            Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed sample data. Columns include all columns in self.gaze, as well as:

            - trialtime : float
                Trial timestamps in milliseconds (since the start of each trial)
            - msgtime : float
                Message timestamps in system timestamps (start time of each trial)
            - msg : str
                Raw message strings (start message of each trial)
            - Additional columns from message parsing if parse_messages=True.
            - Additional columns from self.add_cols if specified.

        Notes
        -----
        Columns are converted to the appropriate data type that supports pd.NA.
        Only trials with both start and stop messages are included in the output.
        """
        # get trial boundaries (start/stop times for each trial)
        messages = self.get_messages()
        
        if messages.empty:
            return pd.DataFrame()
        
        # extract start and stop messages separately
        start_msgs = messages[messages.msg.str.contains(self.start_msg)]
        stop_msgs = messages[messages.msg.str.contains(self.stop_msg)]
        
        # create trial boundaries DataFrame
        # essentially creating a wide format with each row being a trial and the start and stop times
        trial_boundaries = pd.merge(
            start_msgs[['id', 'system_time_stamp']].rename(columns={'system_time_stamp': 'start_time'}),
            stop_msgs[['id', 'system_time_stamp']].rename(columns={'system_time_stamp': 'stop_time'}),
            on='id', how='inner'
        )
        
        if trial_boundaries.empty:
            return pd.DataFrame()
        
        # create a mapping from timestamp to trial_id for all gaze samples
        trial_assignments = []
        
        for _, trial in trial_boundaries.iterrows():
            # find all gaze samples in this trial's time window
            trial_mask = (self.gaze.system_time_stamp >= trial.start_time) & (self.gaze.system_time_stamp < trial.stop_time)
            if trial_mask.any():
                trial_assignments.append({
                    'trial_id': trial.id,
                    'start_time': trial.start_time,
                    'indices': self.gaze[trial_mask].index
                })
        
        if not trial_assignments:
            return pd.DataFrame()
        
        # extract gaze samples and message times
        all_indices = []
        trial_info = []
        msg_times = []
        for assignment in trial_assignments:
            indices = assignment['indices']
            all_indices.extend(indices)
            trial_info.extend([assignment['trial_id']] * len(indices))
            msg_times.extend([assignment['start_time']] * len(indices))

        samples = self.gaze.loc[all_indices].copy()
        samples['trial_id'] = trial_info
        samples['msgtime'] = msg_times
        
        # normalize timestamps
        samples['trialtime'] = ((samples.system_time_stamp - samples.msgtime)/1000).astype(int) # convert to milliseconds
        
        # add message information
        if parse_messages:
            # get start messages for each trial
            start_msg_data = start_msgs.set_index('id')
            message_cols = [col for col in start_msg_data.columns if col not in ['system_time_stamp']]
            
            for col in message_cols:
                samples[col] = samples.trial_id.map(start_msg_data[col])
        else:
            # add raw message
            start_msg_data = start_msgs.set_index('id')
            samples['msg'] = samples.trial_id.map(start_msg_data['msg'])
        
        # add any additional columns
        if self.add_cols:
            samples = samples.assign(**self.add_cols)
        
        # remove the trial_id column from final output
        samples = samples.drop(columns=['trial_id'])
        
        # reset index
        samples = samples.reset_index(drop=True)
        
        return samples.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)