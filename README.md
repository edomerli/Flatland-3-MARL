# 🚂 Flatland-3-MARL
<p align="center">
  <img src="https://i.imgur.com/6jOtX8u.gif">
</p>

Project of the course Multi-Agent Systems of Artificial Intelligence @ University of Bologna, academic year 2023-2024.

In this repo, I implemented MAPPO (Multi-Agent PPO) from scratch and used it on the Multi-Agent [Flatland 3](https://www.aicrowd.com/challenges/flatland-3) challenge.

## ⚙️ Setup
> [!WARNING]
> Flatland works only on Linux, MacOS or Windows using WSL 2.

To create the environment, [install conda](https://www.anaconda.com/download) and then run
```
conda env create -f environment.yml
conda activate flatland-3-marl
```

## 🧠 Train
To train the PPO agent from scratch, run the command
```
python train.py
```
which takes the following command line arguments:
```
usage: train.py [-h] [--env_size ENV_SIZE] [--network_architecture NETWORK_ARCHITECTURE] [--skip_no_choice_cells] [--normalize_v_targets] [--load_checkpoint_env LOAD_CHECKPOINT_ENV] [--use_obs_v1] [--log_video]

optional arguments:
  -h, --help            show this help message and exit
  --env_size ENV_SIZE   The size of the environment to train on. Must be one of [demo, mini, small, medium, large]
  --network_architecture NETWORK_ARCHITECTURE
                        The network architecture to use. Must be one of [MLP, RailTransformer]
  --skip_no_choice_cells
                        Whether to skip cells where the agent has no choice
  --normalize_v_targets
                        Whether to normalize the value targets
  --load_checkpoint_env LOAD_CHECKPOINT_ENV
                        The environment size of the checkpoint to load. Must be one of [demo, mini, small, medium, large]. The latest one with the compatible network_architecture will be loaded from the weights folder.
  --use_obs_v1
                        Whether to use the binary observation v1
  --log_video           Whether to log videos of the episodes to wandb
```

## 🧪 Test (locally)
We evaluated our approach running locally the ``flatland-evaluator`` suite provided by Flatland 3, since submitting the solution for evaluation to AIcrowd always gave a ``failed`` status without enabling debugging (even if debugging was turned on in *aicrowd.json* 🤷‍♂️)

We explain how to do so in this section. You will run the evaluator in a terminal and the repo's solution in another one.

### Terminal 1: evaluator
#### Generate the test cases
First, generate the test cases with the same configuration as [Flatland 3 challenge round 2](https://flatland.aicrowd.com/challenges/flatland3/envconfig.html) using the command below. The generation may take several minutes.

```
python generate_test_cases.py
```

#### Install redis
The Flatland 3 challenge is evaluated in Client/Server architecture, which relies on redis. Please go to https://redis.io/docs/getting-started/ and follow the instructions to install redis.
Check that redis is working properly using:

```
$ redis-cli ping
PONG
```

#### Start the redis evaluator
Note that you must be in the correct conda environment, i.e. the one installed above (N.B. flatland-rl versions 4.X don't have the `flatland-evaluator` command anymore! Make sure you are using a version that has it, e.g. flatland-rl==3.0.15).

```
(flatland-3-marl) $ redis-cli flushall
(flatland-3-marl) $ flatland-evaluator --tests ./env_configs/test/ --shuffle False
```

You should see the evaluator starting up and listening for an agent.

```
(flatland-3-marl) $ flatland-evaluator --tests ./env_configs/test/ --shuffle False
====================
Max pre-planning time: 600      
Max step time: 10
Max overall time: 7200
Max submission startup time: 300
Max consecutive timeouts: 10    
====================
['Test_0/Level_0.pkl', 'Test_0/Level_1.pkl', 'Test_0/Level_2.pkl', 'Test_0/Level_3.pkl', 'Test_0/Level_4.pkl', ...]
['Test_0/Level_0.pkl', 'Test_0/Level_1.pkl', 'Test_0/Level_2.pkl', 'Test_0/Level_3.pkl', 'Test_0/Level_4.pkl', ...]
Listening at :  flatland-rl::FLATLAND_RL_SERVICE_ID::commands
```


### Terminal 2: this repo's solution
Run the submission script:

```
python submission.py --network_architecture MLP --obs_version v2
```


## 🧪 Submit solution to AIcrowd
> [!WARNING]
> We couldn't manage to successfully submit the code for evaluation as the AIcrowd submission would always turn out "failed" after just 2 minutes of running, and couldn't debug it as it wouldn't return any logs anywhere (even though the debug flag is set to true in *aicrowd.json*). We leave below the instructions on how to do so anyway as those might turn out useful for people that wish to try (or want to fix the problem😁)

To submit the code for evaluation to AICrowd:
1. Create a repository on https://gitlab.aicrowd.com/. In our case, we called it "flatland-3" (change the name accordingly in the commands)
2. Clone this repo
3. Push this code to https://gitlab.aicrowd.com/ as a new repo using 
```
cd <your-local-cloned-repo>
# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USERNAME>/flatland-3.git
git push aicrowd
```
4. Create a tag for the submission and push it
```
# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```
5. Visit `gitlab.aicrowd.com/<YOUR_AICROWD_USERNAME>/flatland-3/issues` to see the details of your submission. The submission also appears at https://www.aicrowd.com/challenges/flatland-3/submissions.
