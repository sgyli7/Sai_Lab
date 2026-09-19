"""PPO subclass: per-minibatch KL/lr, and a safe actor unfreeze after critic warmup."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.algorithms import PPO


def _actor_std_mean(policy: Any) -> float | None:
    """Mean exploration std from the learnable parameter, not ``output_std``.

    ``MLPModel.output_std`` reads ``distribution._distribution.stddev``, which
    is only populated after the first stochastic forward. Accessing it before
    ``act()`` raises AttributeError (Python then reports ``no attribute
    'output_std'``). The learnable ``std_param`` exists from construction.
    """
    dist = getattr(policy, "distribution", None)
    if dist is None:
        return None
    param = getattr(dist, "std_param", None)
    if param is not None:
        return float(param.detach().mean().item())
    log_param = getattr(dist, "log_std_param", None)
    if log_param is not None:
        return float(torch.exp(log_param.detach()).mean().item())
    return None


class PPOFinetune(PPO):
    """PPO that records per-minibatch KL and can prime Adam at actor unfreeze.

    After critic-only warmup the actor Adam state is empty, so the first update
    is ≈ ``lr * sign(g)`` on every weight. rsl_rl then retunes lr on *every*
    minibatch (20× per iteration), which either pumps lr on the near-zero first
    KLs or floors it at 1e-5.

    ``mark_unfreeze`` resets actor moments, drops lr, primes Adam with a lr=0
    minibatch, and uses a single epoch so KL vs the rollout cannot compound.
    Later updates pin lr for the whole iteration and apply the adaptive rule
    once on the mean minibatch KL, with a floor at the unfreeze lr.
    """

    last_kl: float = 0.0
    minibatch_kl: list[float]
    minibatch_lr: list[float]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_kl = 0.0
        self.kl_mean = 0.0
        self.kl_max = 0.0
        self.minibatch_kl = []
        self.minibatch_lr = []
        self._prime_actor_once = False
        self._unfreeze_this_update = False
        self._saved_epochs: int | None = None
        self._lr_guard = False
        self._skip_remaining = False
        self.early_stop_minibatches = 0
        self._base_lr = float(self.learning_rate)
        self.min_learning_rate = 1e-5
        self.max_learning_rate = 1e-2
        self._unfreeze_std_before: float | None = None
        self._unfreeze_std_after: float | None = None

    def mark_unfreeze(self, learning_rate: float) -> None:
        """Call once when the actor becomes trainable after critic warmup."""
        self._reset_actor_adam_state()
        lr = float(learning_rate)
        self._set_lr(lr)
        self._prime_actor_once = True
        self._unfreeze_this_update = True
        self._lr_guard = True
        # Keep the 1e-5 floor. Pinning it to unfreeze_lr left 20 minibatches running
        # at 3e-5 while KL-vs-rollout compounded to ~15 (std is not PPO-clipped).
        self.min_learning_rate = 1e-5
        self.max_learning_rate = max(self._base_lr, lr)
        self._saved_epochs = int(self.num_learning_epochs)
        self.num_learning_epochs = 1
        self._unfreeze_std_before = _actor_std_mean(self.get_policy())
        self._unfreeze_std_after = None

    def _reset_actor_adam_state(self) -> None:
        actor_ids = {id(p) for p in self.actor.parameters()}
        state = self.optimizer.state
        for p in list(state.keys()):
            if id(p) in actor_ids:
                del state[p]

    def _set_lr(self, lr: float) -> None:
        self.learning_rate = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _adapt_lr_once(self, mean_kl: float) -> None:
        if self.desired_kl is None:
            return
        lr = float(self.learning_rate)
        if mean_kl > self.desired_kl * 2.0:
            lr = max(self.min_learning_rate, lr / 1.5)
        elif 0.0 < mean_kl < self.desired_kl / 2.0:
            lr = min(self.max_learning_rate, lr * 1.5)
        self._set_lr(lr)

    def update(self) -> dict[str, float]:  # type: ignore[override]
        orig_kl = self.actor.get_kl_divergence
        orig_step = self.optimizer.step
        self.minibatch_kl = []
        self.minibatch_lr = []
        n_step = [0]
        pin = float(self.learning_rate)
        unfreeze_now = bool(self._unfreeze_this_update)
        self._skip_remaining = False
        self.early_stop_minibatches = 0
        self._set_lr(pin)
        kl_stop = None if self.desired_kl is None else float(self.desired_kl) * 2.0

        def _hook(old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
            kl = orig_kl(old_params, new_params)
            k = float(torch.mean(kl).item())
            self.minibatch_kl.append(k)
            self.minibatch_lr.append(float(pin))
            self.last_kl = k
            # KL is vs the rollout distribution, before the step. Further minibatches
            # in this iteration only push it up; skip them once we are past 2× desired.
            if kl_stop is not None and k > kl_stop:
                self._skip_remaining = True
            return kl

        def _step(*args: Any, **kwargs: Any) -> Any:
            n_step[0] += 1
            if self._skip_remaining:
                self.early_stop_minibatches += 1
                return None
            self._set_lr(pin)
            if self._prime_actor_once and n_step[0] == 1:
                saved_lrs = [g["lr"] for g in self.optimizer.param_groups]
                for g in self.optimizer.param_groups:
                    g["lr"] = 0.0
                try:
                    return orig_step(*args, **kwargs)
                finally:
                    for g, lr in zip(self.optimizer.param_groups, saved_lrs):
                        g["lr"] = lr
            return orig_step(*args, **kwargs)

        self.actor.get_kl_divergence = _hook  # type: ignore[method-assign]
        self.optimizer.step = _step  # type: ignore[method-assign]
        try:
            loss_dict: dict[str, Any] = super().update()
        finally:
            self.actor.get_kl_divergence = orig_kl  # type: ignore[method-assign]
            self.optimizer.step = orig_step  # type: ignore[method-assign]
            self._prime_actor_once = False
            if unfreeze_now:
                if self._saved_epochs is not None:
                    self.num_learning_epochs = self._saved_epochs
                    self._saved_epochs = None
                self._unfreeze_this_update = False
                self._unfreeze_std_after = _actor_std_mean(self.get_policy())

        if self.minibatch_kl:
            self.kl_mean = float(sum(self.minibatch_kl) / len(self.minibatch_kl))
            self.kl_max = float(max(self.minibatch_kl))
        else:
            self.kl_mean = float(self.last_kl)
            self.kl_max = float(self.last_kl)
        self.last_kl = self.kl_mean
        loss_dict["kl"] = self.kl_mean
        loss_dict["kl_max"] = self.kl_max
        loss_dict["early_stop_mb"] = float(self.early_stop_minibatches)
        self._set_lr(pin)
        if self._lr_guard and not unfreeze_now:
            self._adapt_lr_once(self.kl_mean)
        return loss_dict
