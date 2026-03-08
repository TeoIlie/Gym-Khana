Installation
============

Gym-Khana is a pure Python package. We recommend installing inside a virtual environment.

Using pip (recommended)
-----------------------

.. code:: bash

   virtualenv gym_env
   source gym_env/bin/activate
   git clone --recurse-submodules https://github.com/TeoIlie/Gym-Khana.git
   cd Gym-Khana
   pip install -e .

Using poetry
------------

.. code:: bash

   poetry install
   source $(poetry env info -p)/bin/activate  # or prefix commands with `poetry run`

Using Docker
------------

A Dockerfile is provided with GUI support via nvidia-docker (NVIDIA GPU required):

.. code:: bash

   docker build -t gymkhana -f Dockerfile .
   docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gymkhana

.. _additional-dependencies:

Additional dependencies
-----------------------

MPC controllers require dependencies that cannot be installed via pip alone. For the reference MPC implementation see the ForzaETH `race_stack <https://github.com/ForzaETH/race_stack>`_.

**acados** (build from source) — see the official `installation docs <https://docs.acados.org/installation/index.html>`_ and `Python interface docs <https://docs.acados.org/python_interface/index.html>`_:

.. code:: bash

   # Clone and build (~/software is only an example install directory)
   git clone https://github.com/acados/acados.git --recurse-submodules ~/software/acados
   cd ~/software/acados && mkdir build && cd build
   cmake -DACADOS_WITH_QPOASES=ON ..
   make install -j$(nproc)

   # Environment variables (add to shell profile)
   export ACADOS_SOURCE_DIR=~/software/acados
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/software/acados/lib

Install the ``acados_template`` Python interface inside your virtual environment:

.. code:: bash

   pip install -e ~/software/acados/interfaces/acados_template
