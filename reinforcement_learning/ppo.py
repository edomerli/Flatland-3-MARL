import torch
import numpy as np
from tqdm import tqdm
import wandb

import torch.bin
from utils.conversions import dict_to_tensor, tensor_to_dict
import utils.global_vars as global_vars
from utils.timer import Timer
from flatland.envs.step_utils.states import TrainState

import yappi

class PPO:
    def __init__(self, actor_critic, env, config, optimizer, scheduler):
        self.actor_critic = actor_critic
        self.env = env
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.iterations_timesteps = config.iteration_timesteps
        self.num_iterations = config.num_iterations

        # episode accumulated rewards
        self.custom_rewards_sum = 0

    def learn(self):
        next_obs, initial_info_dict = self.env.reset()
        # next_obs = torch.tensor(next_obs)
        next_obs = dict_to_tensor(next_obs)
        next_done = False

        for iteration in range(self.num_iterations):
            print(f"=======================Iteration: {iteration}=====================")
            if self.config.profiling:
                yappi.start()

            # next_obs and next_done are returned such that they can be used as the initial state for the next iteration
            observations, actions, log_probs, value_targets, advantages, action_required, next_obs, next_done = self._collect_trajectories(next_obs, next_done, initial_info_dict)

            self._train(observations, actions, log_probs, value_targets, advantages, action_required)

            if self.config.profiling:
                with open("yappi_func_stats.txt", 'w') as f:
                    yappi.get_func_stats().print_all(f)
                with open("yappi_func_stats_tsub_sorted.txt", 'w') as f:
                    yappi.get_func_stats().sort(sort_type="tsub").print_all(f)
                with open("yappi_threads_stats.txt", 'w') as f:
                    yappi.get_thread_stats().print_all(f)

                yappi.stop()
                exit()


    def _collect_trajectories(self, next_obs, next_done, initial_info_dict):
        n_agents = self.env.get_num_agents()

        observations = torch.zeros(self.iterations_timesteps, n_agents, self.config.state_size).to(self.config.device)
        actions = torch.zeros(self.iterations_timesteps, n_agents).to(self.config.device)
        log_probs = torch.zeros(self.iterations_timesteps, n_agents).to(self.config.device)
        custom_rewards = torch.zeros(self.iterations_timesteps).to(self.config.device)
        dones = torch.zeros(self.iterations_timesteps).to(self.config.device)
        values = torch.zeros(self.iterations_timesteps).to(self.config.device)
        advantages = torch.zeros(self.iterations_timesteps).to(self.config.device)

        action_required = torch.zeros(self.iterations_timesteps, n_agents).to(self.config.device)

        # Timers
        inference_timer = Timer()
        env_step_timer = Timer()
        reward_timer = Timer()
        collection_timer = Timer()

        # Actions logging
        actions_count = torch.zeros(self.config.action_size+1)  # +1 for RailEnvActions.DO_NOTHING

        old_info = initial_info_dict

        for step in range(self.iterations_timesteps):
            collection_timer.start()
            if step % (self.iterations_timesteps//10) == 0:
                print(f"Timesteps collected: {step}/{self.iterations_timesteps}")

            # update global step count
            global_vars.global_step += 1

            # load next_obs onto the device
            next_obs = next_obs.to(self.config.device)

            inference_timer.start()
            observations[step] = next_obs
            dones[step] = next_done
            action_required[step] = torch.tensor(list(old_info["action_required"].values())).to(self.config.device)

            with torch.no_grad():
                action, log_prob, _, value = self.actor_critic.action_and_value(next_obs.unsqueeze(0), agents_mask=action_required[step].unsqueeze(0))
                value = value.item()
                action = action.squeeze()
                log_prob = log_prob.squeeze()
            
            actions[step] = action * action_required[step]  # mask the action with action_required, 0 == RailEnvActions.DO_NOTHING
            log_probs[step] = log_prob * action_required[step]  # mask the log_prob with action_required
            values[step] = value
            inference_timer.stop()


            env_step_timer.start()
            # map the action to the agent's action space, but N.B. only for the agents that are required to act!
            action_agent = action + action_required[step]  # mask the action with action_required, 0 == RailEnvActions.DO_NOTHING
            next_obs, reward, done, info = self.env.step(tensor_to_dict(action_agent))
            actions_count += torch.bincount(action_agent.int(), minlength=self.config.action_size+1)
            env_step_timer.stop()


            # compute custom reward and update old_info
            reward_timer.start()
            custom_reward = self.env.custom_reward(done, reward, old_info, info)  # TODO: restore
            # custom_reward = sum(reward.values())  # TODO: remove
            old_info = info
            custom_rewards[step] = custom_reward
            self.custom_rewards_sum += custom_reward
            reward_timer.stop()

            # update next_obs and next_done
            # next_obs = torch.tensor(next_obs)
            next_obs = dict_to_tensor(next_obs)
            # next_done = self.env.is_done(done, info)  # TODO: restore
            next_done = done["__all__"]

            if next_done:
                # print(f"Episode done at step {self.env._elapsed_steps}/{self.env._max_episode_steps}")
                normalized_reward = self.env.normalized_reward(done, reward)

                # N.B. cannot use done dict as a reference as the env sets it to True for all agents even if the environment terminated due to max_steps reached!
                # instead, we have to rely on agent.state
                percentage_done = sum([1 for agent in self.env.agents if agent.state == TrainState.DONE]) / n_agents

                if self.config.wandb:
                    # log real and custom rewards episodically
                    wandb.log({
                        "play/true_episodic_reward": normalized_reward,
                        "play/custom_episodic_reward": self.custom_rewards_sum,
                        "play/percentage_done": percentage_done,
                        "play/episode_length": self.env._elapsed_steps,
                        "play/step": global_vars.global_step
                    })

                # reset the environment
                next_obs, initial_info_dict = self.env.reset()
                # next_obs = torch.tensor(next_obs)
                next_obs = dict_to_tensor(next_obs)
                next_done = False
                self.custom_rewards_sum = 0

            collection_timer.stop()

        if self.config.wandb:
            # log timers
            wandb.log({
                "timer/inference": inference_timer.cumulative_elapsed(),
                "timer/env_step": env_step_timer.cumulative_elapsed(),
                "timer/reward": reward_timer.cumulative_elapsed(),
                "timer/collection": collection_timer.cumulative_elapsed(),
                "timer/step": global_vars.global_step,
            })

            # log actions
            actions_prob = actions_count / max(1, sum(actions_count))
            # renormalize only the non-zero actions
            # this way we have: 
            # - actions_prob[0] = percentage of agents that got masked because not in a decision cell
            # - actions_prob[1:] = policy distribution over the actions
            actions_prob[1:] = actions_prob[1:] / sum(actions_prob[1:])
            wandb.log({
                "action/masked_agent":actions_prob[0],
                "action/left": actions_prob[1],
                "action/forward": actions_prob[2],
                "action/right": actions_prob[3],
                "action/stop": actions_prob[4],
                "action/step": global_vars.global_step
            })

        # GAE
        advantages, value_targets = self._compute_gae(next_obs, next_done, custom_rewards, dones, values)

        if self.config.normalize_v_targets:
            self.actor_critic.update_v_target_stats(value_targets)

        # next_obs and next_done are returned because they will be used as the initial state for the next iteration
        return observations, actions, log_probs, value_targets, advantages, action_required, next_obs, next_done


    @torch.no_grad()
    def _compute_gae(self, next_obs, next_done, rewards, dones, values):
        advantages = torch.zeros_like(rewards).to(self.config.device)
        last_gae = 0

        for t in reversed(range(self.iterations_timesteps)):
            if t == self.iterations_timesteps - 1:
                next_obs = next_obs.to(self.config.device)
                next_value = self.actor_critic.value(next_obs.unsqueeze(0))
                next_nonterminal = 1 - next_done
            else:
                next_value = values[t+1]
                next_nonterminal = 1 - dones[t+1]
            delta = rewards[t] + self.config.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.config.gamma * self.config.lambda_ * next_nonterminal * last_gae
            advantages[t] = last_gae

        value_targets = advantages + values     # Using TD-Lambda as value targets

        return advantages, value_targets

    def _train(self, observations, actions, log_probs, value_targets, advantages, actions_required):
        # TODO remove
        # print(f"Observations: {observations.shape} \n {observations[0]}")
        # print(f"Actions: {actions.shape} \n {actions[0]}")
        # print(f"Log_probs: {log_probs.shape} \n {log_probs[0]}")
        # print(f"Value_targets: {value_targets.shape} \n {value_targets}")
        # print(f"Advantages: {advantages.shape} \n {advantages}")
        # exit()

        # count how many datapoints have at least one agent that is required to act
        n_datapoints = actions_required.sum(-1).nonzero().size(0)
        print(f"Number of datapoints with at least one agent that is required to act: {n_datapoints}/{self.iterations_timesteps}")


        # timers
        train_timer = Timer()
        train_timer.start()

        self.actor_critic.train()   # set the actor_critic to training mode
        epoch_indices = np.arange(self.iterations_timesteps)

        for epoch in tqdm(range(self.config.epochs)):
            np.random.shuffle(epoch_indices)
            for start in range(0, self.iterations_timesteps, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = epoch_indices[start:end]

                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_value_targets = value_targets[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_actions_required = actions_required[batch_indices]

                _, newlogprob, entropy, newvalues = self.actor_critic.action_and_value(batch_observations, action=batch_actions)
                logratio = newlogprob - batch_log_probs
                ratio = torch.exp(logratio)

                # mask the agents that are not required to act
                ratio = ratio * batch_actions_required
                logratio = logratio * batch_actions_required
                entropy = entropy * batch_actions_required

                # compute approx_kl (http://joschu.net/blog/kl-approx.html)
                with torch.no_grad():
                    # mean over the number of real actions taken
                    approx_kl = (((ratio - 1) - logratio) * batch_actions_required).sum() / batch_actions_required.sum()

                # normalize advantages
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                # expand them to have the same value for each agent
                batch_advantages = batch_advantages.unsqueeze(1).expand_as(ratio)
                # TODO (nel caso in cui non mi tornino le performance): WARNING -> More than one element of an expanded tensor 
                # may refer to a single memory location. As a result, in-place operations 
                # (especially ones that are vectorized) may result in incorrect behavior. 
                # If you need to write to the tensors, please clone them first.

                # PPO LOSS
                # Clipped surrogate loss (policy loss)
                loss_pi1 = ratio * batch_advantages
                loss_pi2 = torch.clip(ratio, 1-self.config.eps_clip, 1+self.config.eps_clip) * batch_advantages
                loss_pi = -torch.min(loss_pi1, loss_pi2).mean()
                loss_entropy = entropy.mean()
                loss_policy = loss_pi - self.config.entropy_bonus * loss_entropy

                # MSE loss (value loss)
                loss_value = 0.5 * (newvalues - batch_value_targets).pow(2).mean()

                self.optimizer.zero_grad()
                # TODO: test if doing the following is correct, or I have to do:
                # loss = loss_policy + loss_value
                # loss.backward()
                loss_policy.backward()
                loss_value.backward()
                self.optimizer.step()

                if self.config.wandb and global_vars.global_batch % self.config.batch_log_frequency == 0:
                    wandb.log({
                        "train/loss_pi": loss_pi.item(),
                        "train/loss_v": loss_value.item(),
                        "train/entropy": loss_entropy.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/kl_div": approx_kl.item(),
                        "train/batch": global_vars.global_batch,
                    })

                global_vars.global_batch += 1

            self.scheduler.step()

            if approx_kl > self.config.kl_limit:
                print(f"Early stopping at epoch {epoch} due to KL divergence {round(approx_kl.item(), 4)} > {self.config.kl_limit}")
                break

        if self.config.wandb:
            train_timer.stop()
            wandb.log({"timer/train": train_timer.cumulative_elapsed(), "timer/step": global_vars.global_step})
        
    def save(self, path):
        torch.save(self.actor_critic.policy_state_dict(), path)