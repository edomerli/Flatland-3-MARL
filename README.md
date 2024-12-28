# Flatland-3-MARL
Project of the course Multi-Agent Systems of Artificial Intelligence @ University of Bologna, academic year 2023-2024.

In this repo, I implemented PPO from scratch and used it on the Multi-Agent [Flatland 3](https://www.aicrowd.com/challenges/flatland-3) challenge.

## Setup
To create the environment, [install conda](https://www.anaconda.com/download) and then run
```
conda env create -f environment.yml
conda activate flatland-3-marl
```

## Train
To train the PPO agent from scratch, run the command
```
python train.py
```
which takes the following command line arguments:
```
usage: train.py [-h] [--env_size ENV_SIZE] [--network_architecture NETWORK_ARCHITECTURE] [--skip_no_choice_cells] [--normalize_v_targets] [--load_checkpoint_env LOAD_CHECKPOINT_ENV] [--log_video]

optional arguments:
  -h, --help            show this help message and exit
  --env_size ENV_SIZE   The size of the environment to train on. Must be one of [demo, mini, small, medium, large, huge]
  --network_architecture NETWORK_ARCHITECTURE
                        The network architecture to use. Must be one of [MLP, RailTransformer]
  --skip_no_choice_cells
                        Whether to skip cells where the agent has no choice
  --normalize_v_targets
                        Whether to normalize the value targets
  --load_checkpoint_env LOAD_CHECKPOINT_ENV
                        The environment size of the checkpoint to load. Must be one of [demo, mini, small, medium, large, huge]. The latest one with the compatible network_architecture will be loaded.
  --log_video           Whether to log videos of the episodes to wandb
```

## Test
To test the trained models locally, run
```
TODO
```
To submit the code for evaluation to AICrowd, run
```
TODO
```