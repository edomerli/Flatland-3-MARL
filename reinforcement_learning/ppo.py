import torch
import numpy as np
from tqdm import tqdm
import wandb

from utils.conversions import dict_to_tensor, tensor_to_dict
import utils.global_vars as global_vars

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

    def learn(self):
        next_obs_dict, initial_info_dict = self.env.reset()
        next_obs = dict_to_tensor(next_obs_dict)
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

        observations = torch.zeros(self.iterations_timesteps, n_agents, self.env.obs_builder.observation_dim).to(self.config.device)
        actions = torch.zeros(self.iterations_timesteps, n_agents).to(self.config.device)   # TODO: questo deve essere un one-hot vector? LO E' GIA'? Double check, important
        log_probs = torch.zeros(self.iterations_timesteps, n_agents).to(self.config.device)
        custom_rewards = torch.zeros(self.iterations_timesteps).to(self.config.device)
        dones = torch.zeros(self.iterations_timesteps).to(self.config.device)          # TODO: devo probabilmente tenere anche un vettore di quali siano gli agenti già terminati che sono da "mascherare", oppure posso estendere dones ad avere una dimensione agente
        values = torch.zeros(self.iterations_timesteps).to(self.config.device)
        advantages = torch.zeros(self.iterations_timesteps).to(self.config.device)

        action_required = torch.zeros(self.iterations_timesteps, n_agents).to(self.config.device)

        old_info = initial_info_dict

        last_log_step = 0
        for step in range(self.iterations_timesteps):

            # update global step count
            global_vars.global_step += 1

            # TODO: rimuovi o comunque aggiusta col resto del codice
            # next_obs, old_info = self.env.cycle_until_action_required(next_obs, old_info)

            # load next_obs onto the device
            next_obs = next_obs.to(self.config.device)
            # next_done = next_done.to(self.config.device)    # TODO: needed? non so se serve passare next_done alla rete

            observations[step] = next_obs
            dones[step] = next_done
            action_required[step] = torch.tensor(list(old_info["action_required"].values())).to(self.config.device)

            with torch.no_grad():
                action, log_prob, _, value = self.actor_critic.action_and_value(next_obs.unsqueeze(0), action_required[step].unsqueeze(0))
                value = value.item()
                action = action.squeeze()
                log_prob = log_prob.squeeze()
            
            actions[step] = action * action_required[step]  # mask the action with action_required, 0 = RailEnvActions.DO_NOTHING
            log_probs[step] = log_prob * action_required[step]  # mask the log_prob with action_required
            values[step] = value

            # TODO: remove
            # print(f"Action: {action}, action shape: {action.shape}")
            # print(f"Log_prob: {log_prob}, log_prob shape: {log_prob.shape}")
            # print(f"Value: {value}")

            # print(f"=====================Step: {step}=======================")
            # print(f"Env timestep before step: {self.env._elapsed_steps}")
            # print(f"Old info: {old_info}")
            next_obs, reward, done, info = self.env.step(tensor_to_dict(action)) #TODO: lui usa (action.cpu().numpy())
            # print(f"Next info: {info}")
            # print(f"Env timestep after step: {self.env._elapsed_steps}")


            # compute custom reward and update old_info
            custom_reward = self.env.custom_reward(done, reward, old_info, info)
            old_info = info
            custom_rewards[step] = custom_reward

            # update next_obs and next_done
            next_obs = dict_to_tensor(next_obs)
            # print(f"==================== Step: {step} ====================")
            # print(done)
            next_done = self.env.is_done(done, info)
            # print(next_done)


            if next_done:
                normalized_reward = self.env.normalized_reward(done, reward)

                # log real and custom rewards episodically
                wandb.log({
                    "play/true_episodic_reward": normalized_reward,
                    "play/custom_episodic_reward": custom_rewards[last_log_step:step+1].sum().item(),
                    "play/percentage_done": sum(list(done.values())[:-1]) / n_agents,
                    "play/episode_length": step+1 - last_log_step,
                    "play/step": global_vars.global_step
                })
            
                # reset the environment
                next_obs_dict, initial_info_dict = self.env.reset()
                next_obs = dict_to_tensor(next_obs_dict).to(self.config.device)
                next_done = False

        # GAE
        advantages, value_targets = self._compute_gae(next_obs, next_done, custom_rewards, dones, values)

        # next_obs and next_done are returned because they will be used asthe initial state for the next iteration
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

                _, newlogprob, entropy, newvalues = self.actor_critic.action_and_value(batch_observations, batch_actions)
                logratio = newlogprob - batch_log_probs
                ratio = torch.exp(logratio)

                # mask the agents that are not required to act
                ratio = ratio * batch_actions_required
                logratio = logratio * batch_actions_required
                entropy = entropy * batch_actions_required

                # compute approx_kl (http://joschu.net/blog/kl-approx.html)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

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

                if global_vars.global_batch % self.config.log_frequency == 0:
                    wandb.log({
                        "train/loss_pi": loss_pi.item(),
                        "train/loss_v": loss_value.item(),
                        "train/entropy": loss_entropy.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/batch": global_vars.global_batch
                    })

                global_vars.global_batch += 1

            self.scheduler.step()

            wandb.log({"train/kl_div": approx_kl.item(), "train/batch": global_vars.global_batch})
            if approx_kl > self.config.kl_limit:
                print(f"Early stopping at epoch {epoch} due to KL divergence {round(approx_kl.item(), 4)} > {self.config.kl_limit}")
                break
        


    def save(self):
        pass