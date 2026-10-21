"""Export portable timings and trace aggregates; retain raw traces locally."""
import argparse
import collections
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/profiling'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (args.source / 'training.jsonl').read_text().splitlines()]
    summaries = [s for s in rows if s['event'] == 'summary']
    base = next(s['median_updates_per_second'] for s in summaries if s['case'] == 'fp32-original-host')
    # Input is observation37 + actions175 + time65 (Fourier frequencies1..32).
    # Nine hidden projections (input projection plus eight residual projections).
    dense_weights = (37 + 175 + 65) * 4096 + 8 * 4096**2 + 4096 * 175
    dense_flops = 6 * 1024 * dense_weights
    for row in summaries:
        row['speedup_vs_original'] = row['median_updates_per_second'] / base
        row['dense_matmul_estimated_flops_per_update'] = dense_flops
        row['dense_matmul_estimated_tflops_per_second'] = dense_flops * row['median_updates_per_second'] / 1e12
        # Dense peaks, excluding advertised structured-sparsity doubling.
        peak = 154.8 if row['case'].startswith('bf16') else 38.7 if 'highest' in row['case'] else 77.4
        row['assumed_dense_peak_tflops'] = peak
        row['approximate_dense_mfu'] = row['dense_matmul_estimated_tflops_per_second'] / peak
    traces = []
    for path in sorted((args.source / 'traces').rglob('*.trace.json.gz')):
        events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
        gpu_pids = {x['pid'] for x in events if x.get('name') == 'process_name' and 'GPU' in x.get('args', {}).get('name', '')}
        duration = collections.Counter()
        for x in events:
            if x.get('pid') in gpu_pids and x.get('ph') == 'X':
                duration[x['name']] += x.get('dur', 0)
        total = sum(duration.values())
        traces.append(dict(case=path.relative_to(args.source / 'traces').parts[0],
                           gpu_event_duration_sum_us=total,
                           gemm_event_fraction=sum(v for k, v in duration.items() if any(t in k.lower() for t in ('gemm', 'dot_'))) / total if total else None,
                           top_kernels=[dict(kernel=k, duration_us=v) for k, v in duration.most_common(8)]))
    simulation = []
    sim_log = args.source / 'simulation.log'
    if sim_log.exists():
        for line in sim_log.read_text().splitlines():
            if line.startswith('{'):
                record = json.loads(line)
                if record.get('mode') == 'sim':
                    simulation.append(record)
    result = dict(environment=rows[0], training=summaries, trace_aggregates=traces, simulation=simulation,
                  methodology=dict(dense_flops_formula='6 * batch1024 * (277*4096 + 8*4096^2 + 4096*175)',
                                   dense_weights=dense_weights,
                                   peak_source='https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf',
                                   caveats=['MFU is an approximate dense-matrix estimate, not hardware instruction counters; omits elementwise/normalization/optimizer work and assumes three GEMMs per dense layer.',
                                            'Default-mode MFU uses dense TF32 peak because traces contain tensor-core GEMMs and explicit TF32 has similar throughput; instruction-level precision utilization is not measured.',
                                            'Raw compiler FLOPs omit library/loop work and are not a valid MFU numerator.',
                                            'Repeated synthetic batch excludes dataset sampling and simulator time; no learning-equivalence claim.',
                                            'Simulator timings include per-call setup; this sweep has one measurement per setting.']))
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'training.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    lines = ['# Isolated A6000 profiling', '', '[Results](../../RESULTS.md) · [Exact timings](training.jsonl) · [Analysis and trace aggregates](summary.json)', '',
             'Full 136,179,887-parameter model, batch1024, three timed repeats after warm-up. Raw traces and telemetry remain local.', '',
             '| Case | Median updates/s | Relative speed | Approximate dense MFU |', '| --- | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {r['case']} | {r['median_updates_per_second']:.2f} | {r['speedup_vs_original']:.3f}× | {100*r['approximate_dense_mfu']:.1f}% |")
    lines += ['', 'MFU values use analytical dense-matrix FLOPs and precision-dependent dense peaks from the [NVIDIA GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf). They are estimates, not hardware-counter measurements. The denominator changes with precision, so a faster BF16 case can have lower MFU. Raw XLA cost analysis omits substantial library/loop work and must not be used to infer MFU.', '',
              'Synthetic repeated-batch tests isolate the update loop. They omit real sampling and simulation, and finite losses do not prove task-learning equivalence. The simulator sweep has one observation per setting and includes per-call setup. Larger collections change collection count and are throughput benchmarks only.']
    lines += ['', '![Measured throughput](throughput.png)', '', '| Collection episodes | Parallel cap | Seconds | Episodes/s |', '| ---: | ---: | ---: | ---: |']
    for r in simulation:
        if 'seconds' in r:
            lines.append(f"| {r['episodes']} | {r['parallel']} | {r['seconds']:.2f} | {r['episodes_per_second']:.2f} |")
    tracking = args.output / 'tracking.json'
    if tracking.exists():
        lines += ['', '[W&B profiling comparison](' + json.loads(tracking.read_text())['url'] + ')']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(9, 4))
    ax.barh([r['case'] for r in summaries], [r['median_updates_per_second'] for r in summaries])
    ax.set_xlabel('Model updates / second')
    ax.set_title('Isolated A6000: full model, batch1024, median of 3 repeats')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    figure.tight_layout()
    figure.savefig(args.output / 'throughput.png', dpi=150)
    plt.close(figure)
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
