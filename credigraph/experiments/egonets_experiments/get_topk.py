from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, TypedDict

import numpy as np
from numpy.typing import NDArray
from tgrag.utils.matching import flip_if_needed

DEFAULT_CRAWL = 'CC-MAIN-2025-05'
DEFAULT_PC1_FILE = Path('../../../data/dqr/domain_pc1.csv')


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        'scratch',
        type=Path,
        help='Scratch directory that contains crawl-data/<crawl>/output1/vertices.csv.gz',
    )
    p.add_argument(
        'k',
        nargs='?',
        type=int,
        default=50,
        help='K for top/bottom selection (default: 50)',
    )
    p.add_argument(
        '--crawl',
        type=str,
        default=DEFAULT_CRAWL,
        help=f'Common Crawl name (default: {DEFAULT_CRAWL})',
    )
    p.add_argument(
        '--pc1-file',
        type=Path,
        default=DEFAULT_PC1_FILE,
        help=f'CSV with columns including domain,pc1 (default: {DEFAULT_PC1_FILE})',
    )
    return p.parse_args(argv)


def norm_domain(raw: str) -> str:
    return flip_if_needed(raw.strip().lower())


def load_pc1_map(path: Path) -> dict[str, float]:
    pc1_map: dict[str, float] = {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'PC1 file has no header: {path}')

        domain_key = 'domain'
        pc1_key = 'pc1'
        if domain_key not in reader.fieldnames or pc1_key not in reader.fieldnames:
            raise ValueError(
                f'PC1 file must include columns {domain_key!r} and {pc1_key!r}. '
                f'Found: {reader.fieldnames}'
            )

        for row in reader:
            dom = norm_domain(row[domain_key])
            try:
                pc1 = float(row[pc1_key])
            except (TypeError, ValueError):
                continue
            pc1_map[dom] = pc1
    return pc1_map


def iter_vertices_rows(vertices_gz: Path) -> tuple[list[str], Iterable[list[str]]]:
    f = gzip.open(vertices_gz, 'rt', newline='')
    reader = csv.reader(f)

    try:
        header = next(reader)
    except StopIteration as e:
        f.close()
        raise ValueError(f'Vertices file is empty: {vertices_gz}') from e

    def _rows() -> Iterable[list[str]]:
        try:
            for row in reader:
                yield row
        finally:
            f.close()

    return header, _rows()


def iter_edges(edges_gz: Path) -> Iterable[tuple[str, str]]:
    f = gzip.open(edges_gz, 'rt', newline='')
    reader = csv.DictReader(f)
    try:
        for row in reader:
            yield row['src'], row['dst']
    finally:
        f.close()


def build_neighborhoods(
    edges_gz: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    in_neighbors = defaultdict(set)
    out_neighbors = defaultdict(set)

    for src, dst in iter_edges(edges_gz):
        src = norm_domain(src)
        dst = norm_domain(dst)
        out_neighbors[src].add(dst)
        in_neighbors[dst].add(src)

    return in_neighbors, out_neighbors


class GroupStats(TypedDict):
    in_deg: NDArray[np.int_]
    out_deg: NDArray[np.int_]
    in_pc1: NDArray[np.float_]
    out_pc1: NDArray[np.float_]
    mixing: dict[tuple[str, str, str], int]


def analyze_group(
    nodes: list[str],
    pc1_map: dict[str, float],
    in_nbrs: dict[str, set[str]],
    out_nbrs: dict[str, set[str]],
    top_set: set[str],
    bot_set: set[str],
    label: str,
) -> GroupStats:
    in_degs = []
    out_degs = []
    in_pc1 = []
    out_pc1 = []
    mixing: DefaultDict[tuple[str, str, str], int] = defaultdict(int)

    for n in nodes:
        ins = in_nbrs.get(n, set())
        outs = out_nbrs.get(n, set())

        in_degs.append(len(ins))
        out_degs.append(len(outs))

        for u in ins:
            if u in top_set:
                mixing[(label, 'in', 'top')] += 1
            elif u in bot_set:
                mixing[(label, 'in', 'bottom')] += 1

            if u in pc1_map:
                in_pc1.append(pc1_map[u])

        for v in outs:
            if v in top_set:
                mixing[(label, 'out', 'top')] += 1
            elif v in bot_set:
                mixing[(label, 'out', 'bottom')] += 1

            if v in pc1_map:
                out_pc1.append(pc1_map[v])

    return {
        'in_deg': np.array(in_degs),
        'out_deg': np.array(out_degs),
        'in_pc1': np.array(in_pc1),
        'out_pc1': np.array(out_pc1),
        'mixing': mixing,
    }


def summarize(name: str, stats: GroupStats) -> None:
    print(f'\n=== {name.upper()} ===')

    arr = stats['in_deg']
    print(
        f'in_deg: mean={arr.mean():.2f}, median={np.median(arr):.2f}, '
        f'p25={np.percentile(arr,25):.1f}, p75={np.percentile(arr,75):.1f}'
    )

    arr = stats['out_deg']
    print(
        f'out_deg: mean={arr.mean():.2f}, median={np.median(arr):.2f}, '
        f'p25={np.percentile(arr,25):.1f}, p75={np.percentile(arr,75):.1f}'
    )

    if len(stats['in_pc1']):
        print(f'Incoming neighbor pc1 mean: {stats["in_pc1"].mean():.3f}')
    if len(stats['out_pc1']):
        print(f'Outgoing neighbor pc1 mean: {stats["out_pc1"].mean():.3f}')


def extract_two_hop_subgraph(
    seed: str,
    in_nbrs: dict[str, set[str]],
    out_nbrs: dict[str, set[str]],
) -> set[tuple[str, str]]:
    one_hop = set()
    one_hop |= in_nbrs.get(seed, set())
    one_hop |= out_nbrs.get(seed, set())

    two_hop = set(one_hop)
    for n in one_hop:
        two_hop |= in_nbrs.get(n, set())
        two_hop |= out_nbrs.get(n, set())

    nodes = {seed} | two_hop

    edges = set()
    for u in nodes:
        for v in out_nbrs.get(u, set()):
            if v in nodes:
                edges.add((u, v))
    return edges


def extract_two_hop_labeled(
    seed: str,
    in_nbrs: dict[str, set[str]],
    out_nbrs: dict[str, set[str]],
) -> tuple[dict[str, int], set[tuple[str, str]]]:
    hop = {seed: 0}

    for n in in_nbrs.get(seed, set()) | out_nbrs.get(seed, set()):
        hop[n] = 1

    for n1 in list(hop):
        if hop[n1] == 1:
            for n2 in in_nbrs.get(n1, set()) | out_nbrs.get(n1, set()):
                if n2 not in hop:
                    hop[n2] = 2

    nodes = set(hop.keys())

    edges = set()
    for u in nodes:
        for v in out_nbrs.get(u, set()):
            if v in nodes:
                edges.add((u, v))

    return hop, edges


def write_subgraph(path: Path, edges: set[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'dst'])
        for s, t in sorted(edges):
            w.writerow([s, t])


def write_subgraph_with_hops(
    base_path: Path, hop_map: dict[str, int], edges: set[tuple[str, str]]
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)

    nodes_path = base_path.with_name(base_path.stem + '_nodes.csv')
    edges_path = base_path.with_name(base_path.stem + '_edges.csv')

    with nodes_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'hop'])
        for n, h in sorted(hop_map.items()):
            w.writerow([n, h])

    with edges_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'dst'])
        for s, t in sorted(edges):
            w.writerow([s, t])


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.k <= 0:
        raise ValueError(f'k must be positive; got {args.k}')

    vertices_gz = (
        args.scratch / 'crawl-data' / args.crawl / 'output1' / 'vertices.csv.gz'
    )
    if not vertices_gz.exists():
        raise FileNotFoundError(f'Vertices file not found: {vertices_gz}')

    if not args.pc1_file.exists():
        raise FileNotFoundError(f'PC1 file not found: {args.pc1_file}')

    pc1_map = load_pc1_map(args.pc1_file)

    header, rows_iter = iter_vertices_rows(vertices_gz)

    best_row_by_domain: dict[str, tuple[float, list[str]]] = {}

    for row in rows_iter:
        if not row:
            continue
        raw_dom = row[0]
        dom = norm_domain(raw_dom)
        pc1 = pc1_map.get(dom)
        if pc1 is None:
            continue
        best_row_by_domain.setdefault(dom, (pc1, row))

    top_heap: list[tuple[float, str, list[str]]] = []
    bottom_heap: list[tuple[float, str, list[str]]] = []  # stores (-pc1, dom, row)
    k = args.k

    for dom, (pc1, row) in best_row_by_domain.items():
        top_item = (pc1, dom, row)
        if len(top_heap) < k:
            heapq.heappush(top_heap, top_item)
        elif pc1 > top_heap[0][0]:
            heapq.heapreplace(top_heap, top_item)

        neg_item = (-pc1, dom, row)
        if len(bottom_heap) < k:
            heapq.heappush(bottom_heap, neg_item)
        elif -pc1 > bottom_heap[0][0]:
            heapq.heapreplace(bottom_heap, neg_item)

    topk = sorted(top_heap, key=lambda x: x[0], reverse=True)
    bottomk = sorted(bottom_heap, key=lambda x: x[0])

    top_out = vertices_gz.with_name(f'top_{k}.csv')
    bot_out = vertices_gz.with_name(f'bottom_{k}.csv')

    with top_out.open('w', newline='', encoding='utf-8') as f_top:
        top_writer = csv.writer(f_top, lineterminator='\n')
        top_writer.writerow(['domain', 'pc1', *header[1:]])
        for pc1, dom, row in topk:
            top_writer.writerow([dom, f'{pc1}', *row[1:]])

    with bot_out.open('w', newline='', encoding='utf-8') as f_bot:
        bot_writer = csv.writer(f_bot, lineterminator='\n')
        bot_writer.writerow(['domain', 'pc1', *header[1:]])
        for neg_pc1, dom, row in bottomk:
            pc1 = -neg_pc1
            bot_writer.writerow([dom, f'{pc1}', *row[1:]])

    print(f'[INFO] Wrote top-{k} to {top_out}')
    print(f'[INFO] Wrote bottom-{k} to {bot_out}')

    edges_gz = args.scratch / 'crawl-data' / args.crawl / 'output1' / 'edges.csv.gz'
    in_nbrs, out_nbrs = build_neighborhoods(edges_gz)

    top_nodes = [dom for _, dom, _ in topk]
    bot_nodes = [dom for _, dom, _ in bottomk]

    out_dir = vertices_gz.with_name('ego_2hop')
    out_dir.mkdir(exist_ok=True)

    for group_name, nodes in [('top', top_nodes[:10]), ('bottom', bot_nodes[:10])]:
        for dom in nodes:
            hop_map, edges = extract_two_hop_labeled(dom, in_nbrs, out_nbrs)
            base = out_dir / f"{group_name}_{dom.replace('.', '_')}"
            write_subgraph_with_hops(base, hop_map, edges)

    top_set = set(top_nodes)
    bot_set = set(bot_nodes)

    top_stats = analyze_group(
        top_nodes, pc1_map, in_nbrs, out_nbrs, top_set, bot_set, 'top'
    )
    bot_stats = analyze_group(
        bot_nodes, pc1_map, in_nbrs, out_nbrs, top_set, bot_set, 'bottom'
    )

    summarize('top', top_stats)
    summarize('bottom', bot_stats)

    print('\n=== MIXING ===')
    for k, v in sorted({**top_stats['mixing'], **bot_stats['mixing']}.items()):
        print(k, v)

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
