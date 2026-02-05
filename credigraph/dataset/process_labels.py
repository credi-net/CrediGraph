import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from credigraph.utils.checkers import check_processed_labels
from credigraph.utils.domain_handler import extract_domain

domains = {
    'general': [
        'wikipedia',
    ],
    'misinformation': [
        'misinfo-domains',
        'nelez',
    ],
    'phishing': [
        'phish-and-legit',
        'url-phish',
        'phish-dataset',
        'legit-phish',
    ],
    'malware': [
        'urlhaus',
    ],
}


def process_csv(
    input_csv: Path,
    output_csv: Path,
    is_url: bool,
    domain_col: str,
    label_col: str,
    inverse: bool = False,
    labels: Optional[List] = None,
) -> dict:
    """Process CSV and return stats: {total: count, label_0: count, label_1: count}."""
    domain_labels = defaultdict(list)

    with input_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            domain = row.get(domain_col)
            label = row.get(label_col)

            if not domain or label is None:
                continue

            if is_url:
                domain = extract_domain(domain)

            score = label
            if labels is not None:
                if label == labels[0]:
                    score = 0
                elif label == labels[1]:
                    score = 1
                else:
                    continue

            try:
                domain_labels[domain].append(float(score))
            except ValueError:
                continue

    label_counts = {0: 0, 1: 0}
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, labels in domain_labels.items():
            if not labels:
                continue

            avg_label = sum(labels) / len(labels)
            binary_label = 1 if avg_label >= 0.5 else 0
            if inverse:
                binary_label = 1 - binary_label
            label_counts[binary_label] += 1
            writer.writerow([domain, binary_label])

    check_processed_labels(output_csv)
    return {
        'total': len(domain_labels),
        'label_0': label_counts[0],
        'label_1': label_counts[1],
    }


def process_unlabelled_csv(input_path: Path, output_csv: Path, is_legit: bool) -> dict:
    """Process unlabelled CSV and return stats: {total: count, label_0: count, label_1: count}."""
    label = 1 if is_legit else 0

    domains = set()

    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue

            line = re.sub(r'\s*\(.*?\)\s*$', '', line)
            domain = line.split()[0].lower()

            if domain:
                domains.add(domain)

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain in sorted(domains):
            writer.writerow([domain, label])

    check_processed_labels(output_csv)
    label_0_count = len(domains) if label == 0 else 0
    label_1_count = len(domains) if label == 1 else 0
    return {
        'total': len(domains),
        'label_0': label_0_count,
        'label_1': label_1_count,
    }


def process_goggle(goggle_path: Path, output_csv: Path) -> dict:
    """Process goggle file and return stats: {total: count, label_0: count, label_1: count}."""
    rows = []
    label_counts = {0: 0, 1: 0}

    with goggle_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            if line.startswith('$boost=2'):
                label = 1
            elif line.startswith('$discard'):
                label = 0
            else:
                # ignore $downrank
                continue

            # extract domain
            parts = line.split(',')
            site_part = next((p for p in parts if p.startswith('site=')), None)
            if site_part is None:
                continue

            domain = site_part.split('=', 1)[1]
            rows.append((domain, label))
            label_counts[label] += 1

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        writer.writerows(rows)

    check_processed_labels(output_csv)
    return {
        'total': len(rows),
        'label_0': label_counts[0],
        'label_1': label_counts[1],
    }


def collect_merged(paths: list[Path], output_csv: Path) -> dict[str, list[float]]:
    domain_labels: dict[str, list[float]] = defaultdict(list)

    for csv_path in paths:
        if csv_path.name == output_csv.name:
            continue

        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                domain = row.get('domain')
                if type(domain) == str and domain.startswith('www.'):
                    domain = domain[4:]
                label = row.get('label')

                if not domain or label is None:
                    continue

                try:
                    domain_labels[domain].append(float(label))
                except ValueError:
                    continue

    return domain_labels


def collect_merged_annotated(
    paths: list[Path],
    output_csv: Path,
) -> dict[str, dict[str, float]]:
    """Collect merged labels while tracking which dataset each came from."""
    domain_labels: dict[str, dict[str, float]] = defaultdict(
        dict
    )  # domain -> {dataset_name: score}

    for csv_path in paths:
        if csv_path.name == output_csv.name:
            continue
        # Also skip labels.csv (the merged weak labels)
        if csv_path.name == 'labels.csv':
            continue

        # Extract dataset name from filename (remove .csv)
        dataset_name = csv_path.stem

        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                domain = row.get('domain')
                if type(domain) == str and domain.startswith('www.'):
                    domain = domain[4:]
                label = row.get('label')

                if not domain or label is None:
                    continue

                try:
                    domain_labels[domain][dataset_name] = float(label)
                except ValueError:
                    continue

    return domain_labels


def write_merged(domain_labels: dict[str, list[float]], output_csv: Path) -> None:
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, labels in sorted(domain_labels.items()):
            if not labels:
                continue

            avg_label = sum(labels) / len(labels)
            final_label = 1 if avg_label >= 0.5 else 0
            writer.writerow([domain, final_label])


def write_merged_annotated(
    domain_labels_dict: dict[str, dict[str, float]],
    output_csv: Path,
    dataset_names: list[str],
) -> None:
    """Write labels with one column per dataset."""
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain'] + dataset_names)

        for domain in sorted(domain_labels_dict.keys()):
            labels_dict = domain_labels_dict[domain]
            row: list[str | float] = [domain]
            for dataset_name in dataset_names:
                score = labels_dict.get(dataset_name)
                row.append(score if score is not None else '')
            writer.writerow(row)


def merge_processed_labels(
    processed_dir: Path, output_csv: Path, output_annot_csv: Path
) -> None:
    csv_paths = list(processed_dir.glob('*.csv'))
    # Collect both merged and annotated versions
    domain_labels = collect_merged(csv_paths, output_csv)
    domain_labels_annot = collect_merged_annotated(csv_paths, output_annot_csv)
    # Write both at the same time
    write_merged(domain_labels, output_csv)
    check_processed_labels(output_csv)
    # Get dataset names from annotated data
    dataset_names = sorted(
        set().union(*[d.keys() for d in domain_labels_annot.values()])
    )
    write_merged_annotated(domain_labels_annot, output_annot_csv, dataset_names)
    # Don't check labels_annot.csv as it has different schema with many empty values


def read_weak_labels(path: Path) -> dict[str, int]:
    weak = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get('domain')
            l = row.get('label')
            if not d or l is None:
                continue
            if d.startswith('www.'):
                d = d[4:]
            weak[d] = int(l)
    return weak


def read_reg_scores(path: Path, score_col: str = 'pc1') -> dict[str, float]:
    reg = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get('domain')
            s = row.get(score_col)
            if not d or s is None:
                continue
            if d.startswith('www.'):
                d = d[4:]
            try:
                reg[d] = float(s)
            except ValueError:
                continue
    return reg


def merge_reg_class(
    weak_labels_csv: Path,
    reg_csv: Path,
    output_csv: Path,
) -> None:
    """Final output schema:
    domain, weak_label, reg_score
        - Many domains only have one of the two, then the other is None.
    """
    weak = read_weak_labels(weak_labels_csv)
    reg = read_reg_scores(reg_csv)

    all_domains = sorted(set(weak) | set(reg))

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'weak_label', 'reg_score'])

        for domain in all_domains:
            writer.writerow(
                [
                    domain,
                    weak.get(domain),
                    reg.get(domain),
                ]
            )


def _load_domains(path: Path, domain_col: str = 'domain') -> set:
    domains = set()
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get(domain_col)
            if not d:
                continue
            if d.startswith('www.'):
                d = d[4:]
            domains.add(d)
    return domains


def print_domain_composition(dataset_stats: dict, domains_mapping: dict) -> None:
    """Print a summary table of domain composition per scope with 0-1 breakdown.

    For each scope in domains_mapping (general, misinformation, phishing, malware),
    print the number of domains in each dataset belonging to that scope, with label breakdown.

    Args:
        dataset_stats: dict mapping dataset_name -> {total, label_0, label_1}
        domains_mapping: dict mapping scope -> list of dataset_names
    """
    print('\n======== Domain Composition Summary =========')
    for scope, dataset_names in domains_mapping.items():
        print(f'\n{scope.upper()}:')
        print('-' * 80)

        # Collect counts for datasets in this scope
        print(f"{'Dataset':<30} {'Count':>10} {'Label 0':>10} {'Label 1':>10}")
        total = 0
        total_0 = 0
        total_1 = 0
        for dataset_name in dataset_names:
            stats = dataset_stats.get(
                dataset_name, {'total': 0, 'label_0': 0, 'label_1': 0}
            )
            count = stats['total']
            label_0 = stats['label_0']
            label_1 = stats['label_1']
            print(f'{dataset_name:<30} {count:>10} {label_0:>10} {label_1:>10}')
            total += count
            total_0 += label_0
            total_1 += label_1
        print(f"{'TOTAL':<30} {total:>10} {total_0:>10} {total_1:>10}")


def check_overlaps(strong_labels: Path, weak_labels: Path) -> None:
    """Checks overlaps between used datasets.

    Assumption:
    - The file at `strong_labels` has a path with columns 'domain', 'pc1' (to be changed to 'label' when we merge with other sources than DQR)
    - The file at `weak_labels` has columns 'domain' and 'label', label = 0 for phishing, label = 1 for legitimate

    """
    strong = _load_domains(strong_labels)
    weak = _load_domains(weak_labels)

    overlap = strong & weak
    union = strong | weak

    print(f'# strong: {len(strong)}')
    print(f'# weak: {len(weak)}')
    print(f'# overlap: {len(overlap)}')
    print(f'# union: {len(union)}')


def main() -> None:
    data_dir = Path('./data')
    classification_dir = Path('./data/classification')
    regression_dir = Path('./data/regression')

    class_raw = classification_dir / 'raw'
    class_proc = classification_dir / 'processed'

    regression_dir / 'raw'
    regression_dir / 'processed'

    # Track domain counts per dataset
    dataset_stats = {}

    print('======= LegitPhish ========')
    dataset_stats['legit-phish'] = process_csv(
        Path(f'{class_raw}/url_features_extracted1.csv'),
        Path(f'{class_proc}/legit-phish.csv'),
        is_url=True,
        domain_col='URL',
        label_col='ClassLabel',
        inverse=False,
    )

    print('======= PhishDataset ========')
    dataset_stats['phish-dataset'] = process_csv(
        Path(f'{class_raw}/data_imbal.csv'),
        Path(f'{class_proc}/phish-dataset.csv'),
        is_url=True,
        domain_col='URLs',
        label_col='\ufeffLabels',
        inverse=True,
    )

    print('======= Nelez ========')
    dataset_stats['nelez'] = process_unlabelled_csv(
        Path(f'{class_raw}/dezinformacni_weby (2).csv'),
        Path(f'{class_proc}/nelez.csv'),
        is_legit=False,
    )

    print('======= wiki ========')
    dataset_stats['wikipedia'] = process_goggle(
        Path(f'{class_raw}/wikipedia-reliable-sources.goggle'),
        Path(f'{class_proc}/wikipedia.csv'),
    )

    print('======= URL-Phish ========')
    dataset_stats['url-phish'] = process_csv(
        Path(f'{class_raw}/Dataset.csv'),
        Path(f'{class_proc}/url-phish.csv'),
        is_url=True,
        domain_col='url',
        label_col='label',
        inverse=True,
    )

    print('======== Phish&Legit =======')
    dataset_stats['phish-and-legit'] = process_csv(
        Path(f'{class_raw}/new_data_urls.csv'),
        Path(f'{class_proc}/phish-and-legit.csv'),
        is_url=True,
        domain_col='url',
        label_col='status',
        inverse=False,
    )

    print('======== Misinformation domains =========')
    dataset_stats['misinfo-domains'] = process_csv(
        Path(f'{class_raw}/domain_list_clean.csv'),
        Path(f'{class_proc}/misinfo-domains.csv'),
        is_url=False,
        domain_col='url',
        label_col='type',
        inverse=False,
        labels=['unreliable', 'reliable'],
    )

    print('======== URLhaus =========')
    dataset_stats['urlhaus'] = process_csv(
        Path(f'{class_raw}/urlhaus.csv'),
        Path(f'{class_proc}/urlhaus.csv'),
        is_url=True,
        domain_col='url',
        label_col='threat',
        inverse=False,
        labels=['malware_download', '_unused_'],
    )

    print('======== Merging final labels =========')
    merge_processed_labels(
        class_proc,
        Path(f'{class_proc}/labels.csv'),
        Path(f'{data_dir}/labels_annot.csv'),
    )

    print_domain_composition(dataset_stats, domains)

    check_overlaps(
        Path('./data/dqr/domain_pc1.csv'),
        Path(f'{class_proc}/labels.csv'),
    )

    path = Path('data/labels.csv')

    total = 0

    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

    print(f'Total rows: {total}')


if __name__ == '__main__':
    main()
