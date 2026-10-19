"""Write a balanced experiment queue; one independent job per GPU."""
import json
from pathlib import Path


def matrix():
    jobs = []
    # First compare all four feature settings under identical seeds and data.
    for seed in range(4):
        for feature in ['handcrafted', 'standardized', 'xyz0.1', 'xyz10']:
            jobs.append(dict(run=f'm4-{feature}-s{seed}', mode='streaming', features=feature,
                             env='block-double-task1-v0', seed=seed, episodes=1000))
    for seed in range(4):
        for task, episodes in [('double', 10000), ('single', 10000), ('single', 50000)]:
            jobs.append(dict(run=f'm1-{task}-n{episodes}-s{seed}', mode='fixed', features='handcrafted',
                             env=f'block-{task}-task1-v0', seed=seed, episodes=episodes,
                             data_seed=seed + (10000 if episodes == 50000 else 0)))
    # Start both mysteries immediately, then continue the balanced queue.
    first = next(job for job in jobs if job['run'] == 'm1-double-n10000-s0')
    jobs.remove(first)
    return [first] + jobs


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / 'results' / 'matrix.json'
    target.write_text(json.dumps(matrix(), indent=2) + '\n')
