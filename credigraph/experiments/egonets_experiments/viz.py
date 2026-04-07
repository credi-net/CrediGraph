from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from pyvis.network import Network


def load_egonet(
    nodes_csv: Path, edges_csv: Path
) -> tuple[dict[str, int], pd.DataFrame, Optional[str]]:
    nodes = pd.read_csv(nodes_csv)  # columns: domain, hop
    edges = pd.read_csv(edges_csv)  # columns: src, dst

    hop = dict(zip(nodes['domain'].astype(str), nodes['hop'].astype(int)))

    seed = None
    zero = nodes[nodes['hop'] == 0]
    if len(zero) == 1:
        seed = str(zero.iloc[0]['domain'])

    return hop, edges, seed


def build_pyvis(
    hop: dict[str, int],
    edges: pd.DataFrame,
    seed: Optional[str],
    directed: bool = True,
) -> Network:
    net = Network(height='900px', width='100%', directed=directed, notebook=False)
    net.barnes_hut()

    for node, h in hop.items():
        size = 28 if h == 0 else 18 if h == 1 else 12
        title = f'{node}<br>hop={h}'
        net.add_node(node, label=node, title=title, size=size)

    for _, r in edges.iterrows():
        s = str(r['src'])
        t = str(r['dst'])
        if s not in hop:
            net.add_node(s, label=s, title=f'{s}<br>hop=?')
        if t not in hop:
            net.add_node(t, label=t, title=f'{t}<br>hop=?')
        net.add_edge(s, t)

    net.show_buttons(filter_=['physics'])

    if seed is not None:
        pass

    return net


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--ego-dir',
        type=Path,
        required=True,
        help='Path to ego_2hop directory containing *_nodes.csv and *_edges.csv',
    )
    p.add_argument(
        '--out-dir',
        type=Path,
        default=Path('ego_viz_html'),
        help='Where to write HTML files',
    )
    p.add_argument(
        '--pattern',
        type=str,
        default='*.csv',
        help='Only used to discover files; default picks up all CSVs',
    )
    p.add_argument(
        '--only',
        type=str,
        default='',
        help="Optional substring filter, e.g. 'top_apnews_com' or 'bottom_'",
    )
    p.add_argument(
        '--undirected',
        action='store_true',
        help='Render as undirected (sometimes easier to read)',
    )
    args = p.parse_args()

    ego_dir = args.ego_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    node_files = sorted(ego_dir.glob('*_nodes.csv'))
    if args.only:
        node_files = [f for f in node_files if args.only in f.name]

    if not node_files:
        raise SystemExit(f'No *_nodes.csv files found in {ego_dir}')

    for nodes_csv in node_files:
        base = nodes_csv.name.replace('_nodes.csv', '')
        edges_csv = ego_dir / f'{base}_edges.csv'
        if not edges_csv.exists():
            print(f'[WARN] Missing edges file for {base}: {edges_csv}')
            continue

        hop, edges, seed = load_egonet(nodes_csv, edges_csv)
        net = build_pyvis(hop, edges, seed, directed=(not args.undirected))

        out_html = out_dir / f'{base}.html'
        net.write_html(str(out_html))
        print(f'[OK] Wrote {out_html} (seed={seed})')


if __name__ == '__main__':
    main()
