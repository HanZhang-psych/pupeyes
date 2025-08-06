# PupEyes: Your Buddy for Pupil Size and Eye Movement Data Analysis

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE.md)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pupeyes.readthedocs.io/en/latest/index.html)
[![GitHub issues](https://img.shields.io/github/issues/HanZhang-psych/pupeyes)](https://github.com/HanZhang-psych/pupeyes/issues)
[![Github All Releases](https://img.shields.io/github/downloads/HanZhang-psych/pupeyes/total.svg)]()
[![ReadtheDocs](https://readthedocs.org/projects/pupeyes/badge/?version=latest)](https://pupeyes.readthedocs.io/en/latest/index.html)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HanZhang-psych/pupeyes/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2F)
[![PupEyes](https://raw.githubusercontent.com/HanZhang-psych/pupeyes/refs/heads/main/banner.jpg)](https://pupeyes.readthedocs.io/)

## Overview

PupEyes is a Python package for preprocessing and visualizing eye movement data. It handles pupil size preprocessing and supports interactive visualization of pupil size and fixation data. It was designed to streamline data preparation so you can analyze your data with ease and confidence.

[Try PupEyes in MyBinder!](https://mybinder.org/v2/gh/HanZhang-psych/pupeyes/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2F)

[Tutorials and API Reference](https://pupeyes.readthedocs.io/)

[Preprint](https://osf.io/preprints/psyarxiv/h95ma_v1) 

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

```bash
pip install pupeyes
```

or install the latest development version from Github

```bash
pip install git+https://github.com/HanZhang-psych/pupeyes.git
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

If you use PupEyes in your research, please cite our [preprint](https://osf.io/preprints/psyarxiv/h95ma_v1) 

```bibtex
@misc{zhang_pupeyes_2025,
	title = {{PupEyes}: An Interactive Python Library for Pupil Size and Eye Movement Data Processing},
	url = {https://osf.io/h95ma_v1},
	doi = {10.31234/osf.io/h95ma_v1},
	shorttitle = {{PupEyes}},
	publisher = {{OSF}},
	author = {Zhang, Han and Jonides, John},
	urldate = {2025-04-16},
	date = {2025-04-07}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes in each release.


---------
The project began as an attempt to formalize the eye-tracking processing scripts used in my past research. It then evolved into a much bigger project (as is always the case).

I hope PupEyes will be useful to the eye-tracking community!
