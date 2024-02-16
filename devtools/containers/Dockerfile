# Use the conda-forge base image with Python
FROM mambaorg/micromamba:jammy

LABEL org.opencontainers.image.source=https://github.com/choderalab/asapdiscovery
LABEL org.opencontainers.image.description="ASAPDiscovery toolkit"
LABEL org.opencontainers.image.licenses=MIT

# set environment variables
ENV PYTHONUNBUFFERED 1

RUN micromamba config append channels conda-forge 
RUN micromamba config append channels openeye

COPY --chown=$MAMBA_USER:$MAMBA_USER  devtools/conda-envs/asapdiscovery-ubuntu-latest.yml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/asapdiscovery

RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

 

RUN python -m pip install --no-deps -e asapdiscovery-alchemy
RUN python -m pip install --no-deps -e asapdiscovery-cli
RUN python -m pip install --no-deps -e asapdiscovery-data
RUN python -m pip install --no-deps -e asapdiscovery-dataviz
RUN python -m pip install --no-deps -e asapdiscovery-docking
RUN python -m pip install --no-deps -e asapdiscovery-ml
RUN python -m pip install --no-deps -e asapdiscovery-modeling
RUN python -m pip install --no-deps -e asapdiscovery-simulatio


ARG MAMBA_DOCKERFILE_ACTIVATE=1