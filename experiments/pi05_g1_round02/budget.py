"""Additional eight-hour round, with explicit purpose-driven stage allocation.

Reuses the audited ownership, memory, timeout and actual-usage supervision from
round one. Total additional occupancy stays <= eight hours, including failures.
After fixed teacher collection: 63min diagnosis/data, 222min training,
195min benchmark. Allocation receipts preserve both earlier stage schedules.
"""
from experiments.pi05_g1 import budget as supervisor

LIMITS = {'preflight': 3780, 'training': 13320, 'evaluation': 11700}
TOTAL = 28800


def main():
    if sum(LIMITS.values()) != TOTAL:
        raise RuntimeError('Stage allocation must equal the authorized extra eight hours')
    supervisor.LIMITS = dict(LIMITS)
    supervisor.TOTAL = TOTAL
    supervisor.main()


if __name__ == '__main__':
    main()
