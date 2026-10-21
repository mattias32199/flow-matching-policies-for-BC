"""Export bounded-cache benchmark evidence, separating work savings from wall time."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/cache-performance'))
    args = p.parse_args()
    rows = [json.loads(s) for s in (args.source / 'cache-benchmark.jsonl').read_text().splitlines()]
    old = [r for r in rows if r.get('arm') == 'independent']
    shared = [r for r in rows if r.get('arm') == 'shared']
    assert len(old) == len(shared) == 4, 'Benchmark is not complete.'
    assert len({r['dataset_sha256'] for r in shared}) == 1, 'Shared demonstration hashes differ.'
    assert sum(r['cache_hit'] for r in shared) == 3
    hit = [r for r in shared if r['cache_hit']]
    total = lambda group: sum(r['total_phase_seconds'] for r in group)
    collect = statistics.median(r['collection_seconds'] for r in old)
    training = statistics.median(r['preprocessing_seconds'] + r['training_seconds'] for r in shared)
    read = statistics.median(r['collection_seconds'] for r in hit)
    summary = dict(benchmark=dict(consumers_per_arm=4, updates_per_consumer=old[0]['updates'],
                   precision='float32 network/parameters, default XLA matmul precision',
                   model_parameters=136179887, episodes_per_collection=1000, validation_episodes=128),
                   independent_serial_phase_seconds=total(old), shared_serial_phase_seconds=total(shared),
                   measured_serial_work_speedup=total(old) / total(shared),
                   independent_collection_seconds_median=collect,
                   cache_read_seconds_median=read, cache_read_speedup=collect / read,
                   training_and_preprocessing_seconds_median=training,
                   cache_entry_bytes=shared[0]['cache_entry_bytes'], shared_hashes_identical=True,
                   projected_steady_overlap_speedup=(collect + training) / max(collect, training + read),
                   projected_two_producer_overlap_speedup=(collect + training) / max(collect / 2, training + read),
                   storage_cap_bytes=80000000000,
                   limitations=['Serial four-consumer benchmark measures aggregate work, not four-GPU concurrent wall time.',
                                'Cache-hit latency excludes upstream production; steady-rate overlap projection assumes one additional producer GPU with observed generation rate.',
                                'Current pre-cache workers are unchanged; sustained production throughput awaits the next seed group.',
                                'One measured collection per feature condition; simulator numerics are not bitwise deterministic across fresh generations.'])
    status = next((r for r in reversed(rows) if r.get('event') == 'cache_status'), None)
    if status:
        summary['benchmark_end_cache_status'] = status
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    lines = ['# Shared demonstration cache performance', '', '[Results](../../RESULTS.md) · [Exact measurements](measurements.jsonl) · [Computed analysis](summary.json)', '',
             'One isolated A6000 executed four feature consumers independently, then four sharing one raw collection. Each consumer trained the full model for 1,024 updates at batch 1,024, with unique transition starts and unchanged float32 network computation. Executable warm-up occurred before both arms.', '',
             '| Measurement | Result |', '| --- | ---: |',
             f"| Four independent consumers, summed collection/preprocessing/training | {total(old):.2f}s |",
             f"| Four shared consumers, including one generation | {total(shared):.2f}s |",
             f"| Measured reduction in serial work | {summary['measured_serial_work_speedup']:.2f}× |",
             f'| Median fresh collection | {collect:.2f}s |', f'| Median shared-cache read | {read:.3f}s |',
             f"| Raw collection including validation | {summary['cache_entry_bytes']/1e9:.3f}GB |", '',
             '**The serial-work speedup is not a measured four-GPU wall-time speedup.** Synchronized readers can all wait on one cache miss. A separate producer is required to overlap generation with training. Based on measured collection and training times, the idealized steady overlap improvement with one additional producer GPU is approximately ' + f"{summary['projected_steady_overlap_speedup']:.2f}×; sustained production throughput remains to be measured.", '',
             'All four shared consumers received identical raw-array SHA-256 hashes. Feature transformations remain separate and each run uses its transition starts once. Full-size updates completed with finite losses. Cached entries are deleted after all four variants read them; an 80GB decimal cap covers payloads and in-flight reservations. Existing fixed data is approximately 3.40GB logical (2.5GiB allocated). A full cache falls back to ordinary collection. Producer lookahead is 24 collections; two independent producers are submitted on demand for future seed groups, and it releases its allocation after prolonged budget/no-consumer waits.', '',
             'With two independent producer GPUs, the same idealized overlap model projects ' + f"{summary['projected_two_producer_overlap_speedup']:.2f}×" + ' relative to serial per-run generation. This is a projection; four-way prefill and live production throughput are reported separately.', '',
             'The currently running seed-0 workers retain their original execution. Their checkpoints lack exact sampler state; they were preserved. Future workers load the shared-cache implementation and locally configured producer automatically. No precision, batch-size, update-horizon, or collection-size change was applied.', '',
             '![Measured phase costs](phases.png)', '',
             '## Demonstration scale', '',
             'The [blog](https://seohong.me/blog/behavioral-cloning-mystery/) uses block-double 10K and block-single 10K/50K for Mystery 1; Mystery 4 consumes a fresh stream. Our saved double10K raw state/action dataset is 3.2945GB. With the single-task 1,250-step episode cap, state/action-only upper bounds are approximately 1.86GB for 10K and 9.32GB for 50K. Streaming 2M updates at batch 1,024 uses roughly 2.05B transition starts, corresponding to hundreds of GB cumulatively per seed; the cache is a moving window rather than a complete stored stream.']
    cross = args.source / 'cross-node-read.json'
    if cross.exists():
        (args.output / 'cross-node-read.json').write_text(cross.read_text())
        check = json.loads(cross.read_text())
        lines += ['', f"A reader on the production training node loaded the same entry in {check['read_seconds']:.3f}s with an identical SHA-256; see [cross-node evidence](cross-node-read.json)."]
    tracking = args.output / 'tracking.json'
    if tracking.exists():
        lines += ['', '[W&B performance comparison](' + json.loads(tracking.read_text())['url'] + ')']
    audit = args.output / 'storage-and-privacy-audit.json'
    if audit.exists():
        check = json.loads(audit.read_text())
        lines += ['', '## Storage and private launches', '',
                  f"The completed prefill contains {check['collections']} collections: {check['cache_actual_logical_bytes']/1e9:.2f}GB cached and {check['total_demo_logical_bytes']/1e9:.2f}GB total demonstration data including fixed datasets. Cache accounting matches all manifests, with no remaining write reservations.",
                  'Five submitted jobs were verified through scheduler accounting and local modes: random labels/work paths, owner-only launchers/directories, and an owner-only workspace. Administrators retain access. See the [storage/privacy audit](storage-and-privacy-audit.json).']
    parallel = args.output / 'parallel-prefill-summary.json'
    if parallel.exists():
        check = json.loads(parallel.read_text())
        lines += ['', '## Four-GPU prefill', '',
                  f"Four independent one-GPU jobs completed {check['total_collection_requests']} collection requests: {check['fresh_collections_generated']} fresh generations and {check['cache_reuses']} cache reuses. The start-to-final-finish span was {check['wall_seconds_including_staggered_starts']:.0f}s, including staggered allocation starts. Fresh-generation work summed to {check['summed_fresh_generation_seconds']:.0f}s, yielding {check['generation_work_per_wall_second']:.2f}× serial-work throughput during this phase. This measures collection parallelism, not learner wall time.",
                  '[Exact per-collection records](parallel-prefill.jsonl) · [Parallel analysis](parallel-prefill-summary.json)']
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for column, key, label in [(0, 'collection_seconds', 'Collection/read'), (1, 'preprocessing_seconds', 'Preprocessing'), (2, 'training_seconds', 'Training')]:
        heights = [sum(r[key] for r in group) for group in (old, shared)]
        bottoms = [sum(sum(r[k] for r in group) for k in ('collection_seconds', 'preprocessing_seconds', 'training_seconds')[:column]) for group in (old, shared)]
        ax.bar(['Independent collections', 'Shared collection'], heights, bottom=bottoms, label=label)
    ax.set_ylabel('Summed phase seconds for four consumers')
    ax.set_title('Full-model benchmark: aggregate serial work')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'phases.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
