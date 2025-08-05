# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY pyproject.toml .
COPY poetry.lock .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter and additional notebook dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    matplotlib \
    plotly \
    dash \
    dash-bootstrap-components

# Install PupEyes in development mode
COPY . .
RUN pip install -e .

# Create a non-root user
RUN useradd -m -s /bin/bash jupyter
RUN chown -R jupyter:jupyter /app
USER jupyter

# Expose Jupyter port
EXPOSE 8888

# Set default command to start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"] 