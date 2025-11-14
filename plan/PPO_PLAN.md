# PPO Implementation Plan
To train my drifting controller, I am planning to use **PPO** from SB3 with `wandb` and `tensorboard` integration for model training visualization, and model saving. 

## Requirements
I want the following elements implemented:

### Core functionality

| Status | Task |
|----|----|
| [ ] | PPO |
| [ ] |  Parallelized with 400 simulated instances running concurrently |
| [ ] |  Each rollout consists of 1024 steps with a time step of 0.05s |
| [ ] |  Training with batch size of 1024 |
| [ ] |  Sum of discounted reward is calculated with discount factor gamma=0.99 |
| [ ] |  Learning rate decays from 1 x 10^-3 to 1 x 10^-4 over the course of training |
| [ ] |  120 million time steps |
| [ ] |  Actor and critic networks implemented as MLPs |
| [ ] |  Actor has 2 hidden layers of 256 neurons |
| [ ] |  Critic has two hidden layers of 512 neurons |
| [ ] |  Both actor, critic networks use Leaky Rectified Linear Unit activation function with negative slop 0.2 |

### Code organization and debugging

| Status | Task |
|------|-------|
| [ ] | Integration with **wandb**, including a clean file and folder organization of models, runs, logs, etc, and video recording with `monitor_gym=True` to debug training
| [ ] | Integration with **tensorboard**
| [ ] | Live visualization of training, including key RL metrics such as reward convergence, and NN metrics like `policy_loss` and `value_loss`