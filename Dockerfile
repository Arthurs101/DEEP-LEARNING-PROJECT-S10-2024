FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 

# Update and install required packages
RUN apt-get update --yes && apt-get upgrade --yes
# Update and install required packages, including cairo
RUN apt-get update --yes && apt-get upgrade --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip \
    gdal-bin \
    libgdal-dev \
    r-base \
    python3.10-venv \
    libcairo2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3

# Upgrade pip
RUN pip install --upgrade pip

# Set R_HOME and update PATH
ENV R_HOME=/usr/lib/R
ENV PATH="${R_HOME}/bin:${PATH}"

# Install Anaconda
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh \
    && bash /tmp/anaconda.sh -b -p /anaconda \
    && eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda create --name env \
    && conda activate env

# Create and set the working directory
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app/

# Install Python dependencies with NVIDIA's PyPI URL
RUN pip install --upgrade pip
RUN pip install -i https://pypi.org/simple --extra-index-url https://pypi.nvidia.com -r requirements.txt

# Install additional Python packages like ipywidgets
RUN pip install ipywidgets

# Copy the rest of the application's source code to the container
COPY . /app/

# Expose the port for Jupyter Notebook
EXPOSE 8811

# Command to run Jupyter Notebook and keep the container running
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8811 --no-browser --allow-root && tail -f /dev/null"]
