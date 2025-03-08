# PupEyes: Your Buddy for Pupil Size and Eye Movement Data Analysis

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE.md)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<YOUR URL HERE>)
[![GitHub issues](https://img.shields.io/github/issues/HanZhang-psych/pupeyes/issues)](https://github.com/HanZhang-psych/pupeyes/issues)

## Overview

PupEyes is a Python package for preprocessing and visualizing eye movement data. It handles pupil size preprocessing and supports interactive visualization of pupil size and fixation data. It was designed to streamline data preparation so you can analyze your data with ease and confidence.

**Highlights**

- **Best practices**: The pupil data preprocessing pipeline is desgined based on the best practices available.
- **Pandas integration**: Raw data is cleaned and prepared as a `pandas` dataframe, allowing you to enjoy the vast data analysis and manipulation methods offered by the `pandas` ecosystem.
- **Interactive interface**: Multiple interactive visualizations using `Plotly Dash` allow you understand your data better.

Check out detailed documentation and tutorials here: https://pupeyes.readthedocs.io/.

The project began as an attempt to formalize the eye-tracking processing scripts used in my past research. It then evolved into a much bigger project (as is always the case). While some PupEyes features are specific to Eyelink eye-trackers, many tools are compatible with any eye movement data.

I hope PupEyes will be useful to the eye-tracking community!

## Installation

### Install via `pip`
```bash
# Install the package
pip install pupeyes
```
It's recommended to install PupEyes in a new virtual environment to avoid any potential conflicts with other packages. If you use Anaconda, you can follow these steps:

### Install via `conda`
```bash
# Create a new conda environment called pupeyes-env
conda create -n pupeyes-env python=3.12

# Activate the environment
conda activate pupeyes-env

# Install the package
conda install pupeyes
```

## Documentation

Tutorials and API reference are available at [Read the Docs](https://pupeyes.readthedocs.io/).

## Contributing

Please report bugs if you notice any. See the [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

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
