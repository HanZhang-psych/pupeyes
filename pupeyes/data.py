# -*- coding:utf-8 -*-

"""
Eyelink Data Parsing Module

Author: Han Zhang
Email: hanzh@umich.edu

This module is designed for parsing Eyelink ASC data. It provides functionalities to parse messages, 
samples, fixations, saccades, and blinks.

This module requires specific data formats and is tailored for use with Eyelink eye 
trackers. It is not a general-purpose eye tracking data parser.
"""

import pandas as pd
import numpy as np
from .external.edfreader import read_edf
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
        Message marking the start of a trial
    stop_msg : str 
        Message marking the end of a trial
    msg_format : dict
        Dictionary specifying the format of messages. The messages will be parsed based on this format.
        Example: {'event': str, 'condition': str, 'block': int, 'trial': int}
    delimiter : str
        Character used to separate message components. For example, if messages are formatted as 'event_condition_block_trial',
        the delimiter would be '_'.
    start_notation : str, optional
        Custom notation to mark the start of a trial, if None uses first part of start_msg
    stop_notation : str, optional  
        Custom notation to mark the stop of a trial, if None uses first part of stop_msg
    add_cols : dict, optional
        Additional columns to add to output DataFrames. The dictionary should be in the format {'column_name': column_data}. 
        For example, to add a column 'subject' with value 'S01' to all rows, use {'subject': 'S01'}.

    Attributes
    ----------
    data : pd.DataFrame
        Raw unformatted Eyelink data
    messages : pd.DataFrame
        Extracted messages from the data file

    Methods
    -------
    parse_eyelink_data()
        Loads the raw Eyelink data file
    get_messages()
        Extracts message events from the data
    get_samples(parse_messages=True)
        Retrieves sample data points
    get_fixations(strict=True, parse_messages=True)
        Extracts fixation events
    get_saccades(strict=True, remove_blinks=True, srt=True, parse_messages=True)
        Extracts saccade events
    get_blinks(strict=True, parse_messages=True)
        Extracts blink events
    """

    def __init__(self, path, start_msg, stop_msg, msg_format, delimiter, start_notation=None, stop_notation=None, add_cols= None):
        """
        Initialize EyelinkParser for processing eye tracking data.
        """
        self.path = path
        self.start_msg = start_msg
        self.stop_msg = stop_msg
        self.msg_format = msg_format
        self.delimiter = delimiter

        if start_notation is None:
            self.start_notation = start_msg.split(delimiter)[0] # assuming the first part of the message is the notation
        else:
            self.start_notation = start_notation
        if stop_notation is None:
            self.stop_notation = stop_msg.split(delimiter)[0]
        else:
            self.stop_notation = stop_notation

        self.add_cols = add_cols
        self.data, self.metadata = self.parse_eyelink_data()
        self.messages = self.get_messages()

    
    def parse_eyelink_data(self):
        """
        Loads Eyelink data from a specified file.
        Note that the data doesn't include the line specified as the stop message.
        read_edf is adapted from the pygaze package https://github.com/esdalmaijer/PyGazeAnalyser
        Returns:
            DataFrame: Pandas DataFrame containing the loaded data.
        """
        data, metadata = read_edf(self.path, start=self.start_msg, stop=self.stop_msg) 
        return pd.DataFrame(data), metadata

    
    def get_messages(self):
        """
        Extract MSG rows from Eyelink dataset.

        Returns:
            DataFrame: DataFrame containing message data
        """
        # Extract messages
        msg_list = [(i, time, msg.strip()) 
                   for i, event in enumerate(self.data.events) 
                   if 'msg' in event 
                   for time, msg in event['msg']]
        
        df = pd.DataFrame(msg_list, columns=['id', 'trackertime', 'message'])
    
        # Split messages and assign to result DataFrame
        message_parts = df['message'].str.split(pat=self.delimiter, expand=True)
        for i, col in enumerate(self.msg_format.keys()):
            df[col] = message_parts[i].astype(self.msg_format[col])

        return df.sort_values('trackertime')

            
    def get_samples(self, parse_messages=True):
        """
        Extract raw eye tracking samples from the dataset.

        Parameters
        ----------
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the processed sample data with the following columns:
            - trialtime: Trial timestamps
            - trackertime: Eye tracker timestamps
            - x: X coordinates
            - y: Y coordinates
            - pp: Pupil size measurements (arbitrary unit; measurement unit [area/diameter] depends on recording setting)
            - msg: Raw message strings
            - msgtime: Message timestamps
            Additional columns will be added if parse_messages=True or if self.add_cols is defined.

        Notes
        -----
        All integer columns are converted to float during processing.
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

        # Add id and merge with messages
        #df['id'] = np.repeat(range(len(self.data)), self.data.time.str.len())

        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])
            #df = df.drop(columns=['msg'])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**self.add_cols)

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True) # all integer columns are converted to float

    def get_fixations(self, strict=True, parse_messages=True):
        """
        Extract and process fixation data from the dataset.

        Parameters
        ----------
        strict : bool, optional
            If True, removes fixations that occurred before trial start time (msgtime).
            Default is True.
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing processed fixation data with columns:
            - eye: Eye used for fixation
            - starttime: Start time of fixation
            - endtime: End time of fixation
            - duration: Duration of fixation
            - endx: X-coordinate at end of fixation
            - endy: Y-coordinate at end of fixation
            - msg: Raw message string (if parse_messages=False)
            - msgtime: Message timestamp
            Additional columns are added based on msg_format if parse_messages=True,
            and any additional columns specified in add_cols will be included.
        Notes
        -----
        All integer columns are converted to float during processing.
        """

        # Extract fixations
        s = self.data.events.apply(lambda x: x['Efix']).explode().dropna()
        df = pd.DataFrame(s.tolist(), columns=['eye', 'starttime', 'endtime', 'duration', 'endx', 'endy','msg','msgtime'])
        #df['id'] = s.index
        
        # Remove pre-trial fixations
        if strict:
            df = df[df.starttime.astype(int) >= df.msgtime.astype(int)].reset_index(drop=True)
       
        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])
            #df = df.drop(columns=['msg'])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**(self.add_cols))

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def get_saccades(self, strict=True, remove_blinks=True, srt=True, parse_messages=True):
        """
        Extract and process saccadic eye movements from the dataset.

        Parameters
        ----------
        strict : bool, optional
            If True, removes saccades that occurred before their associated trial 
            message timestamp. Default is True.
        remove_blinks : bool, optional
            If True, removes saccades that overlap with blink periods. Default is True.
        srt : bool, optional
            If True, calculates saccade reaction time (srt) as the difference between 
            saccade start time and message timestamp. Default is True.
        parse_messages : bool, optional
            If True, parses the associated messages according to the predefined 
            message format. Default is True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing processed saccade data with columns:
            - eye: Eye identifier
            - starttime: Start time of saccade
            - endtime: End time of saccade
            - duration: Duration of saccade
            - startx, starty: Starting coordinates
            - endx, endy: Ending coordinates
            - ampl: Amplitude of saccade
            - pv: Peak velocity
            - msg: Associated message (if parse_messages=False)
            - msgtime: Message timestamp
            - srt: Saccade reaction time (if srt=True)
            Additional columns from message parsing if parse_messages=True
            Additional columns from self.add_cols if specified

        Notes
        -----
        - All integer columns are converted to float during processing
        - If remove_blinks=True, saccades overlapping with blinks are removed
        - If strict=True, saccades starting before trial message are removed
        """

        # Get saccades data
        saccades = self.data.events.apply(lambda x: x['Esac']).explode().dropna()
        df = pd.DataFrame(saccades.tolist(), 
                         columns=['eye', 'starttime', 'endtime', 'duration', 
                                 'startx', 'starty', 'endx', 'endy', 'ampl', 'pv','msg','msgtime'])
        #df['id'] = saccades.index
        
        # remove blinks
        if remove_blinks:
            df = self._scrub_blinks(df, self.get_blinks(strict=False))

        # Remove pre-trial saccades
        if strict:
            df = df[df.starttime.astype(int) >= df.msgtime.astype(int)].reset_index(drop=True)

        # compute saccade reaction time
        if srt:
            df['srt'] = df.starttime.astype(int) - df.msgtime.astype(int)
        
        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])
            #df = df.drop(columns=['msg'])

        # Add any additional columns
        if self.add_cols:
            df = df.assign(**(self.add_cols))

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def get_blinks(self, strict=True, parse_messages=True):
        """
        Extract and process blinks data from the dataset.
        Parameters
        ----------
        strict : bool, default=True
            If True, removes blinks that occurred before trial start time (msgtime).
        parse_messages : bool, default=True
            If True, parses the message string into separate columns based on delimiter
            and message format.
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing processed blinks data with the following columns:
            - eye: Eye identifier
            - starttime: Blink start time
            - endtime: Blink end time
            - duration: Blink duration
            - msg: Message string (if parse_messages=False)
            - msgtime: Message timestamp
            Additional columns are added if parse_messages=True, based on msg_format.
        Notes
        -----
        The returned DataFrame's datatypes are automatically optimized using pandas' convert_dtypes.
        """

        blinks = self.data.events.apply(lambda x: x['Eblk']).explode().dropna()
        df = pd.DataFrame(blinks.tolist(), columns=['eye','starttime','endtime','duration','msg','msgtime'])
        
        # Remove pre-trial saccades
        if strict:
            df = df[df.starttime.astype(int) >= df.msgtime.astype(int)].reset_index(drop=True)

        # Split messages and assign to result DataFrame
        if parse_messages:
            message_parts = df['msg'].str.split(pat=self.delimiter, expand=True)
            for i, col in enumerate(self.msg_format.keys()):
                df[col] = message_parts[i].astype(self.msg_format[col])
            #df = df.drop(columns=['msg'])

        return df.convert_dtypes(convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True)

    def _scrub_blinks(self, sac, blk):
        """
        Filter out saccades that overlap with blinks in the dataset.

        Parameters
        ----------
        sac : pandas.DataFrame
            DataFrame containing saccade data with 'starttime' and 'endtime' columns
        blk : pandas.DataFrame
            DataFrame containing blink data with 'starttime' and 'endtime' columns

        Returns
        -------
        pandas.DataFrame
            Filtered saccade DataFrame with blink-overlapping saccades removed,
            index reset to default integer index

        Notes
        -----
        Uses an interval tree for efficient overlap detection between saccades
        and blinks.
        """

        # Create interval tree of blinks
        tree = IntervalTree()
        for _, row in blk.iterrows():
            if row['starttime'] < row['endtime']:
                tree.add(Interval(row['starttime'], row['endtime']))

        # Filter out saccades that overlap with blinks    
        mask = [not tree.overlaps(row['starttime'], row['endtime']) for _, row in sac.iterrows()]
        return sac[mask].reset_index(drop=True)