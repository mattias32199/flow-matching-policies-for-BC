"""Ten scientific experiments; scheduling and allocation are separate choices."""
import json
from pathlib import Path


def matrix():
    jobs = []
    for task, episodes, seed in [('double', 10000, 0), ('double', 10000, 1),
                                  ('single', 10000, 0), ('single', 50000, 0)]:
        jobs.append(dict(run=f'm1-{task}-n{episodes}-s{seed}', mode='fixed',
                         features='handcrafted', env=f'block-{task}-task1-v0',
                         seed=seed, episodes=episodes,
                         data_seed=seed + (10000 if episodes == 50000 else 0)))
    for feature in ['handcrafted', 'standardized', 'xyz0.1', 'xyz10', 'fourier1', 'fourier4']:
        jobs.append(dict(run=f'm4-{feature}-s0', mode='streaming', features=feature,
                         env='block-double-task1-v0', seed=0, episodes=1000))
    return jobs


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / 'results' / 'matrix.json'
    target.write_text(json.dumps(matrix(), indent=2) + '\n')
