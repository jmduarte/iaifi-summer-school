FROM ucsdets/scipy-ml-notebook:2022.3-stable

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN conda install mamba -n base -c conda-forge
RUN mamba install -c pyg -c conda-forge uproot xrootd scikit-learn matplotlib tqdm pyg black

RUN pip install --no-cache-dir mplhep jetnet pydot qkeras==0.9.0 git+https://github.com/fastmachinelearning/hls4ml.git@main#egg=hls4ml[profiling] \
    && pip install --no-cache-dir -U jupyter-book

ADD fix-permissions fix-permissions

RUN chmod +x fix-permissions

RUN fix-permissions /home/$NB_USER

USER $NB_USER
WORKDIR /home/$NB_USER

ENV USER=${NB_USER}
