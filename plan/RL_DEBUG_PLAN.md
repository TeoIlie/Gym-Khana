# RL Training Debug Plan

Currently the drift controller is not training as expected. The following are the top candidates for fixing this issue:

| Status | Issue | Fix |
|--------|-------|-----|
|[ ]| Simple controller not working - PPO trained for 2 mil steps on `train_ppo_example.py` did not converge | Begin with a simple controller, like the original `ppo_example.py` - train this and ensure it works, then iterate: **(1.)** First use single dummy env, then parallelize **(2.)** Go from simple map like `IMS` to challenging map like `Spielberg` **(3.)** Upgrade from `st` to `std` model to validate the model is working |
|[ ]| Incorrect observation normalization | **(1.)** After action normalization, normalizing actions like `steering` and `acceleration` commands may require new boundaries -> `[-1,1]` instead of `[min_steer, max_steer]` **(2.)** Some observations I only guessed their bounds. These need to be validated with command `record_obs_min_max`. For this to work, it must be adjusted to work for a parallelized `SubprocVecEnv`, maybe by creating a custom wrapper that can aggregate the min/max dictionary values across all parallel environments. **(3.)** Use a gym wrapper like `NormalizeObservation` to auto-generate bounds and compare |
|[ ]| Common RL issues | **(1.)** Insufficient steps - add `EvalCallback` with early stoppage condition to train longer until convergence **(2.)** Hyperparameter choice - implement hyperparameter optimization strategies like sweeps **(3.)** Check env with SB3 `check_env` method |
|[ ]| The paper is leaving out some important implementation details, and replication it is not enough | What then...? 🍳 |
|[ ]| Additional unit tests | Test RL config, callbacks, custom activation function, etc. |
