import wandb
import numpy as np

import utils.global_vars as global_vars

class RecorderWrapper():
    def __init__(self, env, episode_frequency_rec):
        """Custom wrapper for recording episodes as videos in wandb.

        Args:
            env (flatland.envs.rail_env.RailEnv): the environment to wrap
            episode_frequency_rec (int): the frequency at which to record episodes
        """
        self.env = env
        self.episode_frequency_rec = episode_frequency_rec

        self.episode_counter = 1
        self.recording = False if episode_frequency_rec > 1 else True
        self.frames = []

    def step(self, action):
        """Overrides env.step() to record episodes as videos in wandb, before forwarding the outputs of the call to the wrapped environment.

        Args:
            action (int): the action to take in the environment
        """
        obs, reward, done_dict, info = self.env.step(action)
        done = done_dict["__all__"]

        image = self.env.render(mode="rgb_array", show=False, return_image=True)
        # put channel as first dimension
        image = np.moveaxis(image, -1, 0)

        if self.recording:
            # record a frame
            self.frames.append(image)
            
            if done:
                # save the whole video recorded so far
                self.save_video()
                self.recording = False
                self.frames = []
        
        if done:
            # count the finished episode and start recording if it's the right time wrt the recording frequency
            self.episode_counter += 1
            if self.episode_counter % self.episode_frequency_rec == 0:
                self.recording = True
            

        return obs, reward, done_dict, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def save_video(self):
        wandb.log({"video": wandb.Video(np.array(self.frames), caption=f"step: {global_vars.global_step} - episode: {self.episode_counter}", fps=30, format="mp4")})

    def close(self):
        super().close()
    