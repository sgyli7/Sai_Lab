"""Train inside the deployed ordinary/sprint controller, including fixed exits.

Ordinary actions are environment dynamics for PPO. They contribute reward and
critic targets across handoffs but must never produce policy-gradient updates.
"""
import numpy as np
import torch
from .models import NativeAnchor


class SprintComposition:
    def __init__(self,path,learn_all=False):
        self.actor=NativeAnchor(path)
        self.learn_all=learn_all
        a=self.actor
        if a.obs_dim!=61 or a.time_input_s or a.heading_input or a.yaw_memory_input or a.state_input or a.task_input:
            raise ValueError('Composed walking requires a frozen ordinary 61D actor')
        self.counts={'learned':0,'ordinary':0}

    @staticmethod
    def mask(worlds):
        return torch.tensor([w.sprint_active() for w in worlds],dtype=torch.bool)

    def actions(self,worlds,proposed):
        result=np.asarray(proposed,np.float32).copy()
        for i,w in enumerate(worlds):
            if self.learn_all or w.sprint_active():self.counts['learned']+=1
            else:
                obs=w.obs()
                if getattr(w,'state_input',''):
                    # The frozen ordinary actor expects body commands, not the
                    # learned residual's velocity/height in those same slots.
                    from sim2sim.obs import build_obs
                    obs=build_obs(w.state,w.last,w.command(),w.home)
                result[i]=self.actor(obs[None])[0]
                self.counts['ordinary']+=1
        return result
