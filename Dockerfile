FROM ucsdets/scipy-ml-notebook:2022.3-stable

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

# Install cmake and XRootD
RUN apt-get update && \
    apt-get upgrade -qq -y && \
    apt-get install -qq -y \
    python3-pip \
    cmake && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*
ADD install_xrootd.sh install_xrootd.sh
RUN bash install_xrootd.sh && \
    rm install_xrootd.sh
ENV PATH /opt/xrootd/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/xrootd/lib

RUN conda install mamba -n base -c conda-forge
RUN mamba install -c pyg -c conda-forge uproot xrootd scikit-learn matplotlib tqdm pyg black

RUN pip install --no-cache-dir mplhep \
    && pip install --no-cache-dir -U jupyter-book


ADD fix-permissions fix-permissions

RUN chmod +x fix-permissions

RUN fix-permissions /home/$NB_USER

USER $NB_USER
WORKDIR /home/$NB_USER

ENV USER=${NB_USER}
