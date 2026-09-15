.. image:: assets/logo.png
   :width: 20
   :align: left
   :alt: logo

Installation
============

Gym-Khana is a pure Python package and requires Python 3.10-3.12.

.. tip::

   We recommend installing inside a virtual environment to avoid dependency conflicts.

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

   git clone --recurse-submodules https://github.com/TeoIlie/Gym-Khana.git
   cd Gym-Khana
   mise install  # mise users only: installs the Python 3.12 pinned in mise.toml
   poetry install --all-groups
   source .venv/bin/activate  # or prefix commands with `poetry run`

Two small config files are tracked in the repository so that this works the same way on every
machine: ``poetry.toml`` places the virtualenv at ``.venv/`` in the project root, and ``mise.toml``
pins Python 3.12.

.. note::

   Check your interpreter with ``python3 --version``. If it falls outside 3.10-3.12 -- common on
   rolling-release distros, which ship a newer default -- install `mise <https://mise.jdx.dev>`_ and
   run ``mise install`` before ``poetry install``. Poetry then resolves the pinned interpreter
   automatically, with no ``poetry env use`` needed. If your ``python3`` is already in range, skip
   that step entirely.

.. note::

   **Distro support.** Nothing here is distro-specific: ``poetry.toml`` and ``mise.toml`` are plain
   config files, and mise installs a prebuilt, glibc-linked interpreter, so ``mise install`` works on
   any mainstream Linux distribution (and on macOS) without a compiler. Two exceptions need mise to
   build Python from source instead -- musl-based systems such as Alpine, and NixOS, where the
   prebuilt binary's dynamic loader is absent. On those, either set ``mise settings python.compile=1``
   and install the usual CPython build dependencies, or skip mise and supply a Python 3.10-3.12 on
   ``PATH`` by other means (the distro's own package, ``pyenv``, ``uv``, conda); poetry only needs a
   matching ``python3``, not mise specifically.

Pre-commit hooks
----------------

``poetry install`` provides ``pre-commit``, but the git hook is wired up once per clone with ``pre-commit install``. After that, ruff runs on staged files at every commit. Run ``pre-commit run --all-files`` to check the whole repo.

.. code:: bash

   pre-commit install

.. _additional-dependencies:

Additional dependencies
-----------------------

.. note::

   MPC controllers require dependencies that cannot be installed via pip alone. These are optional — the core environment and RL training work without them.

For the reference MPC implementation see the ForzaETH `race_stack <https://github.com/ForzaETH/race_stack>`_.

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

.. warning::

   If this fails with ``ModuleNotFoundError: No module named 'vcs_versioning'``, see
   :ref:`acados-vcs-versioning` in Known Issues.

.. tip::

   **VSCode debugging**: If your IDE reports ``ModuleNotFoundError`` for ``acados_template`` or ``casadi`` when debugging, ensure that:

   1. Your virtual environment is selected as the Python interpreter (``Ctrl+Shift+P`` → *Python: Select Interpreter*).
   2. ``acados_template`` is installed into that virtual environment with ``pip install -e`` as shown above (this lets acados use the virtualenv's ``casadi``).
