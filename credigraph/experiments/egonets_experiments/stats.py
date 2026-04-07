from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def resolve_env_path(p: Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


def info(msg: str) -> None:
    print(f'[INFO] {msg}', file=sys.stderr)


def error(msg: str) -> None:
    print(f'[ERROR] {msg}', file=sys.stderr)


def load_topk(path: Path) -> pd.DataFrame:
    rows = []
    mode = None

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('###'):
                mode = 'top' if 'TOP' in line else 'bottom'
                continue
            if line.startswith('domain'):
                continue

            parts = line.split(',')
            if len(parts) != 5 or mode not in ('top', 'bottom'):
                continue

            dom, pc1, ts, in_deg, out_deg = parts
            try:
                rows.append(
                    {
                        'group': mode,
                        'domain': dom.strip(),
                        'pc1': float(pc1),
                        'ts': ts.strip(),
                        'in_deg': int(in_deg),
                        'out_deg': int(out_deg),
                    }
                )
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f'No data parsed from {path}')

    df['total_deg'] = df['in_deg'] + df['out_deg']
    df['in_out_ratio'] = df['in_deg'] / (df['out_deg'] + 1)
    return df


def load_pc1_map(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    if 'domain' not in df.columns or 'pc1' not in df.columns:
        raise RuntimeError(f'PC1 file must have columns domain,pc1: {path}')
    return dict(zip(df.domain.str.lower(), df.pc1))


def get_pc1(dom: str, pc1_map: Dict[str, float]) -> float:
    return pc1_map.get(dom.lower(), np.nan)


def infer_seed_group(base: str) -> Optional[str]:
    if base.startswith('top_'):
        return 'top'
    if base.startswith('bottom_'):
        return 'bottom'
    return None


def base_to_seed_domain(base: str) -> str:
    return base.split('_', 1)[1].replace('_', '.')


def compute_egonet_stats(
    ego_dir: Path,
    pc1_map: Dict[str, float],
    low_pc1: float,
    high_pc1: float,
) -> pd.DataFrame:
    records: List[dict] = []

    paths = sorted(ego_dir.glob('*_nodes.csv'))
    info(f'Found {len(paths)} node files')

    for nodes_path in paths:
        base = nodes_path.name.replace('_nodes.csv', '')
        info(f'Processing {base}')

        edges_path = ego_dir / f'{base}_edges.csv'
        if not edges_path.exists():
            info(f'Skipping {base}: missing edges')
            continue

        seed_group = infer_seed_group(base)
        if seed_group is None:
            info(f'Skipping {base}: cannot infer seed group')
            continue

        seed_domain = base_to_seed_domain(base)

        n_nodes = 0
        pc1_sum = 0.0
        pc1_count = 0

        for chunk in pd.read_csv(nodes_path, usecols=['domain'], chunksize=100_000):
            for dom in chunk['domain']:
                n_nodes += 1
                v = get_pc1(str(dom), pc1_map)
                if not np.isnan(v):
                    pc1_sum += v
                    pc1_count += 1

        mean_neighbor_pc1 = pc1_sum / pc1_count if pc1_count else np.nan
        frac_neighbors_with_pc1 = pc1_count / n_nodes if n_nodes else np.nan

        n_edges = 0
        dst_pc1_sum = 0.0
        dst_pc1_count = 0
        n_low = 0
        n_high = 0

        for chunk in pd.read_csv(edges_path, usecols=['dst'], chunksize=200_000):
            for dst in chunk['dst']:
                n_edges += 1
                v = get_pc1(str(dst), pc1_map)
                if not np.isnan(v):
                    dst_pc1_sum += v
                    dst_pc1_count += 1
                    if v < low_pc1:
                        n_low += 1
                    if v > high_pc1:
                        n_high += 1

        mean_cited_pc1 = dst_pc1_sum / dst_pc1_count if dst_pc1_count else np.nan
        frac_edges_with_pc1 = dst_pc1_count / n_edges if n_edges else np.nan
        frac_edges_to_low = n_low / n_edges if n_edges else np.nan
        frac_edges_to_high = n_high / n_edges if n_edges else np.nan

        records.append(
            {
                'seed': seed_domain,
                'seed_group': seed_group,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'mean_neighbor_pc1': mean_neighbor_pc1,
                'frac_neighbors_with_pc1': frac_neighbors_with_pc1,
                'mean_cited_pc1': mean_cited_pc1,
                'frac_edges_with_pc1': frac_edges_with_pc1,
                'frac_edges_to_low': frac_edges_to_low,
                'frac_edges_to_high': frac_edges_to_high,
            }
        )

    return pd.DataFrame(records)


def summarize_macro(df: pd.DataFrame, group: str) -> dict:
    g = df[df.group == group]
    return {
        'count': len(g),
        'mean_in': g.in_deg.mean(),
        'median_in': g.in_deg.median(),
        'mean_out': g.out_deg.mean(),
        'median_out': g.out_deg.median(),
        'mean_total': g.total_deg.mean(),
        'mean_ratio': g.in_out_ratio.mean(),
        'pc1_mean': g.pc1.mean(),
        'pc1_deg_corr': g.pc1.corr(g.total_deg),
    }


def load_top_bottom(top_path: Path, bottom_path: Path) -> pd.DataFrame:
    top = pd.read_csv(top_path)
    bottom = pd.read_csv(bottom_path)

    top['group'] = 'top'
    bottom['group'] = 'bottom'

    df = pd.concat([top, bottom], ignore_index=True)

    required = {'domain', 'pc1', 'in_deg', 'out_deg', 'group'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Missing columns in top/bottom CSVs: {missing}')

    df['total_deg'] = df['in_deg'] + df['out_deg']
    df['in_out_ratio'] = df['in_deg'] / (df['out_deg'] + 1)

    return df


def print_macro(df: pd.DataFrame) -> None:
    print('\n=== MACRO SUMMARY ===')
    for grp in ['top', 'bottom']:
        s = summarize_macro(df, grp)
        print(f'\n{grp.upper()}:')
        for k, v in s.items():
            print(f'  {k:18s}: {v:.3f}' if isinstance(v, float) else f'  {k:18s}: {v}')


def print_egonet_summary(egos: pd.DataFrame, low_pc1: float, high_pc1: float) -> None:
    print('\n=== EGO-NET CREDIBILITY MIXING ===')
    for grp in ['top', 'bottom']:
        g = egos[egos.seed_group == grp]
        print(f'\n{grp.upper()} seeds:')
        print(f'  egonets: {len(g)}')
        print(f'  mean nodes: {g.n_nodes.mean():.1f}')
        print(f'  mean edges: {g.n_edges.mean():.1f}')
        print(f'  mean neighbor pc1: {g.mean_neighbor_pc1.mean():.3f}')
        print(f'  mean cited pc1: {g.mean_cited_pc1.mean():.3f}')
        print(f'  frac edges to LOW (<{low_pc1}): {g.frac_edges_to_low.mean():.3f}')
        print(f'  frac edges to HIGH (>{high_pc1}): {g.frac_edges_to_high.mean():.3f}')
        print(f'  coverage nodes: {g.frac_neighbors_with_pc1.mean():.3f}')
        print(f'  coverage edges: {g.frac_edges_with_pc1.mean():.3f}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--top', default='top_50.csv')
    p.add_argument('--bottom', default='bottom_50.csv')
    p.add_argument('--pc1', default='../../../data/dqr/domain_pc1.csv')
    p.add_argument(
        '--ego-dir', default='$SCRATCH/crawl-data/CC-MAIN-2025-05/output1/ego_2hop'
    )
    p.add_argument('--low', type=float, default=0.3)
    p.add_argument('--high', type=float, default=0.7)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    pc1_path = resolve_env_path(Path(args.pc1))
    ego_dir = resolve_env_path(Path(args.ego_dir))

    if not pc1_path.exists():
        error(f'Missing pc1 file: {pc1_path}')
        return 1
    if not ego_dir.exists():
        error(f'Missing ego dir: {ego_dir}')
        return 1

    info(f'PC1:  {pc1_path}')
    info(f'Egos: {ego_dir}')

    script_dir = Path(__file__).parent

    top_path = resolve_env_path(script_dir / args.top)
    bottom_path = resolve_env_path(script_dir / args.bottom)

    if not top_path.exists():
        error(f'Missing top file: {top_path}')
        return 1
    if not bottom_path.exists():
        error(f'Missing bottom file: {bottom_path}')
        return 1

    info(f'Top:    {top_path}')
    info(f'Bottom: {bottom_path}')

    df = load_top_bottom(top_path, bottom_path)
    print_macro(df)

    pc1_map = load_pc1_map(pc1_path)
    egos = compute_egonet_stats(ego_dir, pc1_map, args.low, args.high)

    if egos.empty:
        error('No egonets loaded.')
        return 1

    print_egonet_summary(egos, args.low, args.high)
    return 0


if __name__ == '__main__':
    sys.exit(main())
