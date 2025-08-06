# PupEyes: Your Buddy for Pupil Size and Eye Movement Data Analysis

PupEyes is a Python package for preprocessing and visualizing eye movement data. It handles pupil size preprocessing and supports interactive visualization of pupil size and fixation data. It supports both Eyelink and Tobii data as well as any generic eye-tracking dataset that conforms to minimal formatting requirements. 

## Installation

```bash
pip install pupeyes
```

or install the latest development version from Github

```bash
pip install git+https://github.com/HanZhang-psych/pupeyes.git
```

## Executable Notebooks

[Try PupEyes in MyBinder!](https://mybinder.org/v2/gh/HanZhang-psych/pupeyes/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2F)

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


## Tutorials

```{tableofcontents}
```
## Issues?

If you encounter a problem while running the code, or have suggestions for improvement, please [submit an issue here](https://github.com/HanZhang-psych/pupeyes/issues).

