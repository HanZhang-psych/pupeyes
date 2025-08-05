# PupEyes Docker Setup

This guide explains how to run PupEyes using Docker, so you can run the tutorials in a virtual environment.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Docker Desktop** - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
2. **Docker Compose** - Usually comes with Docker Desktop

## Quick Start

1. **Clone or download the PupEyes repository**
   ```bash
   git clone https://github.com/HanZhang-psych/pupeyes.git
   cd pupeyes
   ```

2. **Start the container**
   ```bash
   ./start-pupeyes.sh start
   ```

3. **Access Jupyter Lab**
   - Open your web browser
   - Go to: http://localhost:8888
   - You'll see Jupyter Lab with all PupEyes notebooks available

4. **Stop the container**
   ```bash
   ./start-pupeyes.sh stop
   ```

## What's Included

The Docker container includes:

- **Python 3.11** with all PupEyes dependencies
- **Jupyter Lab** for running notebooks
- **All PupEyes packages** installed and ready to use
- **Sample data** from the `docs/data/` directory
- **Interactive applications** (Pupil Viewer, Fixation Viewer, AOI Drawer)

## Available Notebooks

Once you start the container, you'll have access to these notebooks in the `docs/` directory:

- `read_data_eyelink.ipynb` - Reading Eyelink data
- `read_data_tobii_titta.ipynb` - Reading Tobii Titta data  
- `read_data_tobii_prolab.ipynb` - Reading Tobii Pro Lab data
- `pupil_preproc.ipynb` - Pupil preprocessing
- `pupil_stats.ipynb` - Pupil statistics
- `fixation_viewer.ipynb` - Fixation visualization
- `aoi_analysis.ipynb` - Area of Interest analysis

## Issues

If you encounter issues:

1. Check the [PupEyes documentation](https://pupeyes.readthedocs.io/)
2. Report issues on [GitHub](https://github.com/HanZhang-psych/pupeyes/issues)
3. Check Docker logs: `./start-pupeyes.sh logs`

## License

This Docker setup follows the same license as PupEyes (GNU General Public License v3.0). 