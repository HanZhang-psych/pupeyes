# -*- coding:utf-8 -*-

"""
Eyelink Data Parsing Module

This module is designed for parsing Eyelink ASC data. It provides functionalities to parse messages, 
samples, fixations, saccades, and blinks.
"""

import pandas as pd
import numpy as np
from ..external.edfreader import read_edf
from intervaltree import Interval, IntervalTree

class EyelinkReader:
    """
    A class to read and parse Eyelink eye tracking data files.
    This class handles loading and parsing of Eyelink data files, providing methods to extract messages,
    samples, fixations, saccades and blinks. It supports customizable message formats and additional
    column specifications.

    Parameters
    ----------
    path : str
        Path to the Eyelink data file
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
    data : pd.DataFrame
        Raw unformatted Eyelink data
    messages : pd.DataFrame
        Extracted messages from the data file
    metadata : dict
        Metadata from the Eyelink data file

    Examples
    --------
    >>> reader = EyelinkReader(
    ...     path='subject01.asc',
    ...     start_msg='TRIAL_START',
    ...     stop_msg='TRIAL_END',
    ...     msg_format={'marker': str, 'event': str, 'block': int, 'trial': int},
    ...     delimiter=' '
    ... )
    """

    def __init__(self, path, start_msg, stop_msg, msg_format, delimiter, add_cols=None, progress_bar=True):
        """
        Initialize EyelinkReader for processing eye tracking data.

        Parameters
        ----------
        path : str
            Path to the Eyelink data file
        start_msg : str
            Common part of message marking the start of a trial. For example: 'TRIAL_START'
        stop_msg : str 
            Common part of message marking the end of a trial. For example: 'TRIAL_END'
        msg_format : dict
            Dictionary specifying the format of event markers. For example:
            {'marker': str, 'event': str, 'block': int, 'trial': int}
        delimiter : str
            Character used to separate message components. For example: ' '
        add_cols : dict, optional
            Additional columns to add to output DataFrames. For example:
            {'subject': 'S01', 'session': 1}
        progress_bar : bool, optional
            If True, shows a progress bar while reading the data file. Default is True.
        """
        self.path = path
        self.start_msg = start_msg
        self.stop_msg = stop_msg
        self.msg_format = msg_format
        self.delimiter = delimiter
        self.add_cols = add_cols
        self.data, self.metadata = self.parse_eyelink_data(progress_bar)
        self.messages = self.get_messages()

    def parse_eyelink_data(self, progress_bar):
        """
        Loads and parses raw Eyelink data from the specified file. A wrapper for read_edf function.

        This method reads the Eyelink data file and extracts both the data and metadata.

        Returns
        -------
        tuple
            A tuple containing:

            - pd.DataFrame
                The parsed Eyelink data
            - dict
                Metadata from the Eyelink file

        Notes
        -----
        The read_edf function is adapted from the pygaze package 
        https://github.com/esdalmaijer/PyGazeAnalyser
        """
        data, metadata = read_edf(self.path, start=self.start_msg, stop=self.stop_msg, progress_bar=progress_bar) 
        return pd.DataFrame(data), metadata

    def get_messages(self):
        """
        Extract and process marker events from the Eyelink dataset.

        This method extracts all message events from the data and parses them according
        to the specified message format and delimiter.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed message data with columns:

            - id : int
                Trial identifier
            - trackertime : float
                Eye tracker timestamps
            - message : str
                Raw message string
            - Additional columns to store parsed message parts based on msg_format specification.
            - Additional columns from self.add_cols are added if specified.

        Notes
        -----
        Messages are split using the specified delimiter and parsed according
        to the data types specified in msg_format.
        """
        msg_list = [(i, time, msg.strip()) 
                   for i, event in enumerate(self.data.events) 
                   if 'msg' in event 
                   for time, msg in event['msg']]
        
        df = pd.DataFrame(msg_list, columns=['id', 'trackertime', 'message'])
    
        message_parts = df['message'].str.split(pat=self.delimiter, expand=True)
        for i, col in enumerate(self.msg_format.keys()):
            df[col] = message_parts[i].astype(self.msg_format[col])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**(self.add_cols))

        # sort by trackertime
        return df.sort_values('trackertime')

    def get_samples(self, parse_messages=True):
        """
        Extract and process raw eye tracking samples from the dataset.

        This method extracts all sample data points from the Eyelink recording,
        including gaze position, pupil size, and associated messages.

        Parameters
        ----------
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed sample data with columns:

            - trialtime : float
                Trial timestamps
            - trackertime : float
                Eye tracker timestamps
            - x : float
                X coordinates of gaze position
            - y : float
                Y coordinates of gaze position
            - pp : float
                Pupil size measurements (arbitrary unit; measurement unit [area/diameter] 
                depends on recording setting)
            - msg : str
                Raw message strings
            - msgtime : float
                Message timestamps
            - Additional columns from message parsing if parse_messages=True.
            - Additional columns from self.add_cols if specified.

        Notes
        -----
        Columns are converted to the appropriate data type that supports pd.NA.
        """

        # Create dataframe from sample data
        df = pd.DataFrame({
            'trialtime': np.concatenate(self.data.time),
            'trackertime': np.concatenate(self.data.trackertime), 
            'x': np.concatenate(self.data.x),
            'y': np.concatenate(self.data.y),
            'pp': np.concatenate(self.data['size']),
            'msg': np.concatenate(self.data.last_msg),
            'msgtime': np.concatenate(self.data.last_msg_time),
        })

        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**self.add_cols)

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def get_fixations(self, strict=True, parse_messages=True):
        """
        Extract and process fixation events from the dataset.

        This method extracts all fixation events from the Eyelink recording,
        including their duration, position, and associated messages.

        Parameters
        ----------
        strict : bool, optional
            If True, removes "bridge" fixations: fixations with start times before an event and end times after the same event.
            Default is True.
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed fixation data with columns:

            - eye : str
                Eye identifier (left/right)
            - starttime : float
                Start time of fixation
            - endtime : float
                End time of fixation
            - duration : float
                Duration of fixation in milliseconds
            - endx : float
                X-coordinate at end of fixation
            - endy : float
                Y-coordinate at end of fixation
            - msg : str
                Raw message string (if parse_messages=False)
            - msgtime : float
                Message timestamp
            - Additional columns from message parsing if parse_messages=True.
            - Additional columns from self.add_cols if specified.

        Notes
        -----
        Columns are converted to the appropriate data type that supports pd.NA.
        If strict=True, fixations starting before their associated trial message
        are removed from the output.
        """

        # Extract fixations
        s = self.data.events.apply(lambda x: x['Efix']).explode().dropna()
        df = pd.DataFrame(s.tolist(), columns=['eye', 'starttime', 'endtime', 'duration', 'endx', 'endy','msg','msgtime'])
        
        # Remove pre-trial fixations
        if strict:
            df = df[df.starttime.astype(float) >= df.msgtime.astype(float)].reset_index(drop=True)
       
        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**(self.add_cols))

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def get_saccades(self, strict=True, remove_blinks=True, srt=True, parse_messages=True):
        """
        Extract and process saccadic eye movements from the dataset.

        This method extracts all saccade information from the dataset,
        with the option to remove saccades that overlap with blinks and calculate
        saccade reaction times.

        Parameters
        ----------
        strict : bool, optional
            If True, removes "bridge" saccades: saccades with start times before an event and end times after the same event.
            Default is True.
        remove_blinks : bool, optional
            If True, removes saccades that overlap with blink periods. This is recommended for Eyelink data as Eyelink embeds a blink inside a saccade.
            Default is True.
        srt : bool, optional
            If True, calculates saccade reaction time (srt) as the difference between 
            saccade start time and message timestamp. Default is True.
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed saccade data with columns:

            - eye : str
                Eye identifier (left/right)
            - starttime : float
                Start time of saccade
            - endtime : float
                End time of saccade
            - duration : float
                Duration of saccade in milliseconds
            - startx : float
                Starting X coordinate
            - starty : float
                Starting Y coordinate
            - endx : float
                Ending X coordinate
            - endy : float
                Ending Y coordinate
            - ampl : float
                Amplitude of saccade in degrees
            - pv : float
                Peak velocity in degrees/second
            - msg : str
                Associated message (if parse_messages=False)
            - msgtime : float
                Message timestamp
            - srt : float
                Saccade reaction time (if srt=True)
            - Additional columns from message parsing if parse_messages=True.
            - Additional columns from self.add_cols if specified.

        Notes
        -----
        - Columns are converted to the appropriate data type that supports pd.NA.
        - If remove_blinks=True, saccades overlapping with blinks are removed
        - If strict=True, saccades starting before trial message are removed
        - Saccade reaction time (srt) is calculated as starttime - msgtime
        """

        # Get saccades data
        saccades = self.data.events.apply(lambda x: x['Esac']).explode().dropna()
        df = pd.DataFrame(saccades.tolist(), 
                         columns=['eye', 'starttime', 'endtime', 'duration', 
                                 'startx', 'starty', 'endx', 'endy', 'ampl', 'pv','msg','msgtime'])
        
        # remove blinks
        if remove_blinks:
            df = self._scrub_blinks(df, self.get_blinks(strict=False))

        # Remove pre-trial saccades
        if strict:
            df = df[df.starttime.astype(float) >= df.msgtime.astype(float)].reset_index(drop=True)

        # compute saccade reaction time
        if srt:
            df['srt'] = df.starttime.astype(float) - df.msgtime.astype(float)
        
        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**(self.add_cols))

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def get_blinks(self, strict=True, parse_messages=True):
        """
        Extract and process blink events from the dataset.

        This method extracts all blink events from the Eyelink recording,
        including their duration and associated messages.

        Parameters
        ----------
        strict : bool, optional
            If True, removes "bridge" blinks: blinks with start times before an event and end times after the same event.
            Default is True.
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed blink data with columns:

            - eye : str
                Eye identifier (left/right)
            - starttime : float
                Start time of blink
            - endtime : float
                End time of blink
            - duration : float
                Duration of blink in milliseconds
            - msg : str
                Message string (if parse_messages=False)
            - msgtime : float
                Message timestamp
            - Additional columns from message parsing if parse_messages=True.
            - Additional columns from self.add_cols if specified.

        Notes
        -----
        - Columns are converted to the appropriate data type that supports pd.NA.
        - Blinks are detected by Eyelink's algorithm.
        """
        blinks = self.data.events.apply(lambda x: x['Eblk']).explode().dropna()
        df = pd.DataFrame(blinks.tolist(), columns=['eye','starttime','endtime','duration','msg','msgtime'])
        
        # Remove pre-trial saccades
        if strict:
            df = df[df.starttime.astype(float) >= df.msgtime.astype(float)].reset_index(drop=True)

        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def _scrub_blinks(self, sac, blk):
        """
        Filter out saccades that overlap with blinks in the dataset.

        This method creates an interval tree from blink periods and removes any
        saccades that overlap with these periods.

        Parameters
        ----------
        sac : pd.DataFrame
            DataFrame containing saccade data with 'starttime' and 'endtime' columns
        blk : pd.DataFrame
            DataFrame containing blink data with 'starttime' and 'endtime' columns

        Returns
        -------
        pd.DataFrame
            Filtered saccade DataFrame with blink-overlapping saccades removed,
            with index reset to default integer index.

        Notes
        -----
        Uses an interval tree for efficient overlap detection between saccades
        and blinks. Saccades that start or end during a blink period are removed.
        """

        # Create interval tree of blinks
        tree = IntervalTree()
        for _, row in blk.iterrows():
            if row['starttime'] < row['endtime']:
                tree.add(Interval(row['starttime'], row['endtime']))

        # Filter out saccades that overlap with blinks    
        mask = [not tree.overlaps(row['starttime'], row['endtime']) for _, row in sac.iterrows()]
        return sac[mask].reset_index(drop=True)