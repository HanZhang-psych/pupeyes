# PupEyes: Empowering Your Pupillometry and Eye Movement Data Processing

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE.md)

## Overview

PupEyes is a Python package for processing and visualizing eye movement data. It provides comprehensive tools for pupil size preprocessing and fixation visualization. Its interactive tools allow you to explore your data with ease.

Check out the tutorials here: https://pupeyes.readthedocs.io/.

The project began as an attempt to formalize the eye-tracking processing scripts used in my past research. It then evolved into a much bigger project with all the data visualization tools. 

I hope PupEyes will be useful to those who are interested in eye-tracking.

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
