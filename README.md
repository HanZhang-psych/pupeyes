# PupEyes: A Python Library for Pupillometry and Eye Movement Processing

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE.md)

## Overview

PupEyes is a Python library designed for processing, analyzing, and visualizing pupillometry and eye movement data. It provides functionalities for processing pupil sizes (e.g., deblinking, artifact rejection, smoothing, interpolating, and baseline correction) as well as fixations/saccades. Its interactive tools allow you to explore the data with ease.

The package is an attempt to formalize and share the eye-tracking processing scripts used in my research. I hope this will help those who would like to learn more about eye-tracking.

## Installation

### Install via `pip`
```bash
# Install the package
pip install pupeyes
```
It's recommended to install PupEyes in a new virtual environment to avoid any potential conflicts with other packages. If you use Anaconda, you can follow these steps:

### Install via `conda`
```bash
# Create a new conda environment
conda create -n pupeyes-env python=3.12

# Activate the environment
conda activate pupeyes-env

# Install the package
pip install pupeyes
```

## Quick Start

### Read Raw Data

```python
import pupeyes as pe

# your asc file should contain custom messages that mark trials, e.g.,
# start of block A trial 6 --> start A 6
# end of block A trial 6 --> end A 6
msg_format = {'event':str, 'block':str, 'trial':int}
delimiter = ' '

# parse ASC data
raw = EyelinkReader('path/to/data.asc', start_msg='start', stop_msg = 'end', msg_format=msg_format, delimiter=delimiter)

# get raw gaze samples
samples = raw.get_samples()

# get fixations
fixations = raw.get_fixations()

# get saccades
saccades = raw.get_saccades()
```

Currently only Eyelink .ASC files are supported. But as long as you can read in a dataframe with necessary columns (e.g., `timestamp`, `x`, `y`, `size`, and column(s) to identify trials), you can still use the package to analyze your data. 

### Pupil Preprocessing

```python
import pupeyes as pe

# Load and process pupil data
pupil_data = pe.PupilData.from_file("your_data.csv")
processed_data = pupil_data.process()

# Analyze saccades
saccades = pe.detect_saccades(processed_data)

# Define and analyze Areas of Interest
aoi = pe.AOI.from_coordinates([(x1, y1), (x2, y2)])
aoi_metrics = aoi.analyze(processed_data)

# Visualize results
pe.plot_pupil_trace(processed_data)
```

## Documentation

Detailed documentation is available at [Read the Docs](https://pupeyes.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## Code of Conduct

This project adheres to a [Code of Conduct](CONDUCT.md). By participating, you are expected to uphold this code.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use PupEyes in your research, please cite:

```bibtex
@software{pupeyes2024,
  author = {Zhang, Han},
  title = {PupEyes: A Python Library for Pupillometry and Eye Movement Processing},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/pupeyes}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes in each release.

## Contact

For questions and support, please [open an issue](https://github.com/yourusername/pupeyes/issues) on our GitHub repository.
