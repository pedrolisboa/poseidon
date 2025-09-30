# Use an official PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:22.10-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies for code-server
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    sudo \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y \
    # libc6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    
RUN wget https://github.com/coder/code-server/releases/download/v4.8.3/code-server_4.8.3_amd64.deb && \
dpkg -i code-server_4.8.3_amd64.deb && \
rm -f code-server_4.8.3_amd64.deb

# Install Jupyter and its necessary dependencies
RUN pip install jupyterlab

# # Expose Jupyter's default port
# EXPOSE 10001
# # Expose code-server port
# EXPOSE 10000

RUN code-server --install-extension ms-python.python

RUN wget https://github.com/microsoft/vscode-cpptools/releases/download/v1.9.8/cpptools-linux.vsix && \
    code-server --install-extension cpptools-linux.vsix && \
    rm -f cpptools-linux.vsix

COPY requirements.txt .

# Install any necessary dependencies specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
