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