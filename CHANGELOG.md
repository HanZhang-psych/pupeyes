# Changelog

<!--next-version-placeholder-->

## v0.1.0 (03/08/2025)

- First release of `pupeyes`!

## v0.2.0 (3/22/2025)

- Use strictly True of False for baseline_outlier and trace_ouliter columns in the summary data to avoid pandas ambiguous interpretation of NA in a boolean context.
- Removed some redundant dependencies and relaxed Python version restrictions to hopefully speed up resolving dependencies during install.
- Added plot_gaze_surface() to check gaze position.
- Improved summary stats for filter_position().
- Updated docs to demonstrate plot_gaze_surface()

## v0.2.1 (3/22/2025)

- A better fix for how NA is handled in check_baseline_outliers()
- Added save option for plot_gaze_surface()

## v0.2.2 (4/3/2025)

- Added ipywidgets as dependency

## v0.3.0 (8/5/2025)

- Added Tobii data support with TobiiTittaReader for hdf5 format data
- Added upsample and downsample functions to pupil preprocessing pipeline
- Refactored sampling frequency check to handle non-Eyelink trackers
- More friendly messages for new column names during pupil preprocessing
- Add warning for non-integer timestamp columns for PupilProcessor
- Add support for specifying different missing pupil values during initialization
- Added h5py and tables as dependency for Tobii data file handling
- Restructured documentation to include dedicated Tobii data sections
- Updated API reference to include Tobii data support
- Removed deprecated read_data notebook and replaced with device-specific notebooks
- Minor changes to tutorials

## v0.3.1 (8/5/2025)
- Minor bug fixes.

## v0.3.2 (8/5/2025)
- Automatically convert dtypes for pupil size data when creating PupilProcessor object.