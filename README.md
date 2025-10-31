![Python 3.10-3.12](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
![Code Style](https://github.com/f1tenth/f1tenth_gym/actions/workflows/lint.yml/badge.svg)

# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
virtualenv gym_env
source gym_env/bin/activate
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

Alternatively, install using poetry:

```bash
poetry install
source $(poetry env info -p)/bin/activate # or instead of sourcing, prefix commands with `poetry run`
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Wanbd
The wandb models are available here: https://wandb.ai/teo-altum-quinque-queen-s-university/projects 

## Formatting/Linting
Run formatting mannually with `black .`. Linting also runs automatically due to settings in `.vscode/settings.json`.

## Debugging
1. Debug with breakpoints by looping through environment steps, as in `tests/drift_observation_test.py`
2. `gym.make()` configurations:
  1. Run with `render_mode` set to `human` to visualize the process
  2. Set `"render_track_lines": True` (it is `False` by default) to render the centerline in **green** and the raceline in **red**
  3. Set `"render_lookahead_curvatures": True` (it is `False` by default) to visualize lookahead curvature sampling points ahead of the vehicle in **yellow**. Optional parameters:
     - `"lookahead_n_points": 10` - Number of lookahead points (default: 10)
     - `"lookahead_ds": 0.3` - Spacing between points in meters (default: 0.3m)
  4. Set `"debug_frenet_projection" = True` to visualize the Frenet coordinates are correct
  5. Set `"normalize"` to `True/False` for normalizing the observation space. Only `"drift"` observation type currently is able to be normalized 
  6. Set `"record_obs_min_max"` to `True/False` to record min/max observation values during training, and tweak normalization bounds defined in `utils.py::calculate_norm_bounds`

## Important files:
* `f1tenth_gym/envs/base_classes.py:503` defines the `step` method. 
  * notice that the action space is defined as an `ndarray` with 
    1. the first element being desired **steering angle**
    2. second element is desired **velocity**.
* dynamics models are defined in `f1tenth_gym/envs/dynamic_models`
  * `single_track.py` models the single-track dynamics model, but no tires
  * `multi_body.py` models the car in far more detail, including tires, but may be overkill with RL
  * to replicate the paper - single-track dynamics + Pacejka tire model - it may be necessary to write a custom hybrid approach using `single_track.py` and `multi_body.py`

## Branches and the f1tenth_gym fork
* The original `f1tenth_gym` project has branch `main` which in this project is renamed to `f1tenth_main_original`, and `rl_example`, which in this project is renamed to `main`
* This is bc the `rl_example` contains all the code I am actively using to build this project

## Tire parameters
* Parameters for the 1/10 scale f1tenth car to be used with the `STD` model are defined in `f110_env.py` as `f1tenth_std_vehicle_params`. They are created as a mix of existing f1tenth params and tire parameters adjusted from the fullscale car. 
* In future I may measure these parameters from real data for more accurate fitting
* To maintain a history of parameter choices, and how they compare with the correct behaviour on the fullscale car, tests script `tests/model_validation/test_f1tenth_std_params.py` creates comparison figures along with parameter YAML file dump ordered by date created inside folder `figures/tire_params`

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
