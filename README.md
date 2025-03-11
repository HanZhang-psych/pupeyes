# PupEyes: Your Buddy for Pupil Size and Eye Movement Data Analysis

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE.md)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pupeyes.readthedocs.io/en/latest/intro.html)
[![GitHub issues](https://img.shields.io/github/issues/HanZhang-psych/pupeyes/issues)](https://github.com/HanZhang-psych/pupeyes/issues)
[![Github All Releases](https://img.shields.io/github/downloads/HanZhang-psych/pupeyes/total.svg)]()
![](https://raw.githubusercontent.com/HanZhang-psych/pupeyes/refs/heads/main/banner.jpg)

## Overview

PupEyes is a Python package for preprocessing and visualizing eye movement data. It handles pupil size preprocessing and supports interactive visualization of pupil size and fixation data. It was designed to streamline data preparation so you can analyze your data with ease and confidence.

## Higlights

### Best practices 

The pupil data preprocessing pipeline is desgined based on the best practices available.

### Pandas integration

Raw data is cleaned and prepared as a `pandas` dataframe, allowing you to enjoy the vast data analysis and manipulation methods offered by the `pandas` ecosystem.

### Interactive interface

Multiple interactive visualizations using `Plotly Dash` allow you understand your data better.

**Pupil Viewer**: Examining Pupil Preprocessing Steps

![](https://raw.githubusercontent.com/HanZhang-psych/pupeyes/refs/heads/main/docs/assets/pupil_viewer.gif)


**Fixation Viewer**: Visualize Fixation Patterns

![](https://raw.githubusercontent.com/HanZhang-psych/pupeyes/refs/heads/main/docs/assets/fixation_viewer.gif)


**AOI Drawing Tool**: Flexibly Define AOIs

![](https://raw.githubusercontent.com/HanZhang-psych/pupeyes/refs/heads/main/docs/assets/aoi_drawer.gif)


## Installation

### Install via `pip`
```bash
pip install pupeyes
```

### Install via `conda`
It's recommended to install PupEyes in a new virtual environment to avoid any potential conflicts with other packages. If you use Anaconda, you can follow these steps:

```bash
conda create -n pupeyes-env python=3.12
conda activate pupeyes-env
```

This will create a new conda environment called pupeyes-env and activates it. Now, you can install PupEyes:

```bash
conda install pupeyes
```

## Documentation

Tutorials and API reference are available at [Read the Docs](https://pupeyes.readthedocs.io/).

## Compatibility

While some PupEyes features are specific to Eyelink eye-trackers, many tools are compatible with any eye movement data.

## Contributing

Please report bugs if you notice any. See the [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use PupEyes in your research, please cite:

```bibtex
@software{pupeyes2025,
  author = {Zhang, Han},
  title = {PupEyes: Your Buddy for Pupil Size and Eye Movement Data Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HanZhang-psych/pupeyes}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes in each release.


---------
The project began as an attempt to formalize the eye-tracking processing scripts used in my past research. It then evolved into a much bigger project (as is always the case).

I hope PupEyes will be useful to the eye-tracking community!