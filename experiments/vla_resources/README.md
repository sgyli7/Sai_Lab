# Shared GB10 queue

Both chats are peers. Each finite GPU stage obtains a FIFO ticket, then holds
`.scratch/vla_gpu_coordination/gb10-exclusive.lock` around its own budget
supervisor. CPU preparation can run concurrently. Waiting starts no supervisor
and changes no GPU budget ledger. Existing artifacts and foreign processes remain
owned by their originating chat.

From the Sai_Lab root:

```bash
python3 -m experiments.vla_resources.gpu_queue request \
  --owner YOUR_THREAD_ID --stage preflight --max-seconds 900
# Use the returned ticket. The inner command MUST have matching --ledger and
# --stage, and an explicit --max-seconds no greater than the ticket duration.
python3 -m experiments.vla_resources.gpu_queue run \
  --ticket t000001 --ledger /absolute/your/gpu_budget.json -- \
  python3 -m YOUR_BUDGET_MODULE --ledger /absolute/your/gpu_budget.json \
  --stage preflight --max-seconds 900 -- YOUR_WORKLOAD
python3 -m experiments.vla_resources.gpu_queue status
```

The original legacy barrier is cleared only by its owner after audited cleanup.
A completed supervisor must add exactly one ledger row proving inactive/failed
owned unit, closed accounting, complete Docker cleanup, and no pending launcher.
Failure with proved cleanup releases the turn. Missing/uncertain cleanup blocks
subsequent tickets, including after wrapper death; it requires owner audit.
No automatic stale-ticket removal or takeover exists. A waiting ticket can be
cancelled only using its owner id. `run --wait-seconds N` returns 75 on wait timeout
and leaves the ticket waiting. Do not leave an unused waiting ticket at the head.

A signal is forwarded only to the wrapper's own supervisor child. The wrapper
waits for its ledger cleanup evidence. A bounded supervisor remains mandatory;
the queue itself does not implement GPU memory or runtime limits.

Validation: eight CPU subprocess tests cover FIFO/exclusion, barrier/unchanged
ledger while waiting, duplicate launch, owner cancellation, duration admission,
failed-but-cleaned release, uncertain cleanup, and signal forwarding/cleanup.
The abandoned Grok implementation is preserved in the scratch coordination root;
the reviewed smaller implementation supersedes it.
