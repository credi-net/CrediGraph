import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from credigraph.utils.checkers import check_processed_labels, is_valid_domain
from credigraph.utils.domain_handler import extract_domain
from credigraph.utils.mergers import merge_processed_labels, merge_reg_class
from credigraph.utils.readers import read_reg_scores, read_weak_labels


def process_csv(
    input_csv: Path,
    output_csv: Path,
    is_url: bool,
    domain_col: str,
    label_col: str,
    inverse: bool = False,
    labels: Optional[List] = None,
) -> Dict[str, float]:
    """Process a labeled CSV into a domain-level binary label file.

    Aggregates labels per domain, averages them, thresholds at 0.5 to form a
    binary label, and optionally inverts the result.

    Parameters:
        input_csv : Path
            Path to the input CSV file.
        output_csv : Path
            Path where the processed CSV will be written.
        is_url : bool
            Whether the domain column contains full URLs that must be normalized.
        domain_col : str
            Name of the column containing domains or URLs.
        label_col : str
            Name of the column containing labels.
        inverse : bool, optional
            Whether to invert the final binary label.
        labels : Optional[List], optional
            Optional mapping for categorical labels, e.g. [negative_label, positive_label].

    Returns:
        Dict[str, float]
            Mapping from domain to average label (before binary conversion).
    """
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

            if not is_valid_domain(domain):
                continue

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

    domain_averages = {}

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, label_list in domain_labels.items():
            if not label_list:
                continue

            avg_label = sum(label_list) / len(label_list)
            domain_averages[domain] = avg_label
            binary_label = 1 if avg_label >= 0.5 else 0
            if inverse:
                binary_label = 1 - binary_label
            writer.writerow([domain, binary_label])

    check_processed_labels(output_csv)
    return {
        domain: avg for domain, avg in domain_averages.items() if domain is not None
    }


def process_unlabelled_csv(
    input_path: Path, output_csv: Path, is_legit: bool
) -> Dict[str, float]:
    """Process an unlabeled domain list into a labeled CSV, all with same label depending on ``is_legit``.

    Parameters:
        input_path : Path
            Path to the input text or CSV file containing domains.
        output_csv : Path
            Path where the processed CSV will be written.
        is_legit : bool
            If True, assigns label 1; otherwise assigns label 0.

    Returns:
        Dict[str, float]
            Mapping from domain to its label (as float).
    """
    label = 1.0 if is_legit else 0.0

    domains = set()

    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue

            line = re.sub(r'\s*\(.*?\)\s*$', '', line)
            domain = line.split()[0].lower()

            if not is_valid_domain(domain):
                continue

            if domain:
                domains.add(domain)

    domain_scores = {domain: label for domain in domains}

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain in sorted(domains):
            writer.writerow([domain, int(label)])

    check_processed_labels(output_csv)
    return domain_scores


def process_goggle(goggle_path: Path, output_csv: Path) -> Dict[str, float]:
    """Process a .goggle rules file into a domain-level label CSV.

    Parameters:
        goggle_path : Path
            Path to the .goggle configuration file.
        output_csv : Path
            Path where the processed CSV will be written.

    Returns:
        Dict[str, float]
            Mapping from domain to its label (as float).
    """
    rows = []
    domain_scores = {}

    with goggle_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            if line.startswith('$boost=2'):
                label = 1.0
            elif line.startswith('$discard'):
                label = 0.0
            else:
                # ignore $downrank
                continue

            # extract domain
            parts = line.split(',')
            site_part = next((p for p in parts if p.startswith('site=')), None)
            if site_part is None:
                continue

            domain = site_part.split('=', 1)[1]
            rows.append((domain, int(label)))
            domain_scores[domain] = label

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        writer.writerows(rows)

    check_processed_labels(output_csv)
    return domain_scores


def process_urlhaus(urlhaus_path: Path, output_csv: Path) -> Dict[str, float]:
    """Process URLHaus malware database into a domain-level label CSV.

    Marks domains with 'malware_download' threat as unreliable (0).
    Extracts domain names from URLs.

    Parameters:
        urlhaus_path : Path
            Path to the URLHaus CSV file.
        output_csv : Path
            Path where the processed CSV will be written.

    Returns:
        Dict[str, float]
            Mapping from domain to its label (as float).
    """
    domains = set()

    with urlhaus_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row in reader:
            url = row.get('url')
            threat = row.get('threat')

            if not url or threat is None:
                continue

            if threat != 'malware_download':
                continue

            domain = extract_domain(url)
            if not is_valid_domain(domain):
                continue
            if domain:
                domains.add(domain)

    domain_scores = {domain: 0.0 for domain in domains}

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain in sorted(domains):
            writer.writerow([domain, 0])

    check_processed_labels(output_csv)
    return domain_scores


def create_merged_annotated(
    final_domains: List[str],
    dataset_sources: Dict[str, Dict[str, float]],
    output_csv: Path,
    weak_labels: Dict[str, int],
    reg_scores: Dict[str, float],
) -> None:
    """Create an annotation CSV tracking original source labels for each domain.

    Parameters:
        dataset_sources : Dict[str, Dict[str, float]]
            Mapping from dataset name to mapping of domain to original score.
        output_csv : Path
            Path where the annotation CSV will be written.
        weak_labels : Optional[Dict[str, int]], optional
            Final aggregated weak labels per domain.
        reg_scores : Optional[Dict[str, float]], optional
            Regression scores per domain.
    """
    all_domains: Set[str] = set()
    for domain_map in dataset_sources.values():
        for domain in domain_map.keys():
            if domain and is_valid_domain(domain):
                all_domains.add(domain)

    if weak_labels:
        all_domains.update(d for d in weak_labels if is_valid_domain(d))

    if reg_scores:
        all_domains.update(d for d in reg_scores if is_valid_domain(d))

    all_domains_list = sorted(all_domains)
    dataset_names = sorted(dataset_sources.keys())

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        extra_cols: List[str] = []
        if weak_labels is not None:
            extra_cols.append('weak_label')
        if reg_scores is not None:
            extra_cols.append('reg_score')

        header = ['domain'] + dataset_names + extra_cols
        writer.writerow(header)

        for domain in all_domains_list:
            row = [domain]
            for dataset_name in dataset_names:
                score = dataset_sources[dataset_name].get(domain, '')
                row.append(str(score))
            if weak_labels is not None:
                row.append(str(weak_labels.get(domain, '')))
            if reg_scores is not None:
                row.append(str(reg_scores.get(domain, '')))
            writer.writerow(row)

    print(f'[INFO] Created annotations CSV: {output_csv}')
    print(f'[INFO]   Total domains: {len(all_domains)}')
    print(f'[INFO]   Datasets tracked: {len(dataset_names)}')


def create_labels_csv(
    dataset_sources: Dict[str, Dict[str, float]],
    reg_scores: Dict[str, float],
    output_csv: Path,
) -> None:
    """Create final labels.csv with:
    - domain
    - credibility (regression average)
    - reliability (binary, thresholded using mean regression score).
    """
    # --- collect all domains ---
    all_domains: Set[str] = set()
    for src in dataset_sources.values():
        all_domains.update(d for d in src if is_valid_domain(d))
    all_domains.update(d for d in reg_scores if is_valid_domain(d))

    # --- regression credibility ---
    credibility: Dict[str, float] = {}
    for domain, score in reg_scores.items():
        if is_valid_domain(domain):
            credibility[domain] = float(score)

    # --- global threshold from regression ---
    if credibility:
        threshold = sum(credibility.values()) / len(credibility)
    else:
        raise ValueError('No regression scores available to compute threshold.')

    # --- classification reliability ---
    reliability: Dict[str, int] = {}
    for domain in all_domains:
        cls_scores: List[float] = []
        for src in dataset_sources.values():
            if domain in src:
                cls_scores.append(src[domain])

        if not cls_scores:
            continue

        avg_cls = sum(cls_scores) / len(cls_scores)
        reliability[domain] = 1 if avg_cls >= threshold else 0

    # --- write CSV ---
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'credibility', 'reliability'])

        for domain in sorted(all_domains):
            writer.writerow(
                [
                    domain,
                    credibility.get(domain, ''),
                    reliability.get(domain, ''),
                ]
            )

    print(f'[INFO] Created labels CSV: {output_csv}')
    print(f'[INFO]   Threshold (mean regression): {threshold:.4f}')
    print(f'[INFO]   Total domains: {len(all_domains)}')


def main() -> None:
    data_dir = Path('./data')
    classification_dir = Path('./data/classification')
    regression_dir = Path('./data/regression')

    class_raw = classification_dir / 'raw'
    class_proc = classification_dir / 'processed'

    regression_dir / 'raw'
    reg_proc = regression_dir / 'processed'
    dataset_sources: Dict[str, Dict[str, float]] = {}

    print('======= LegitPhish ========')
    dataset_sources['legit-phish'] = process_csv(
        Path(f'{class_raw}/url_features_extracted1.csv'),
        Path(f'{class_proc}/legit-phish.csv'),
        is_url=True,
        domain_col='URL',
        label_col='ClassLabel',
        inverse=False,
    )

    print('======= PhishDataset ========')
    dataset_sources['phish-dataset'] = process_csv(
        Path(f'{class_raw}/data_imbal.csv'),
        Path(f'{class_proc}/phish-dataset.csv'),
        is_url=True,
        domain_col='URLs',
        label_col='\ufeffLabels',
        inverse=True,
    )

    print('======= Nelez ========')
    dataset_sources['nelez'] = process_unlabelled_csv(
        Path(f'{class_raw}/dezinformacni_weby (2).csv'),
        Path(f'{class_proc}/nelez.csv'),
        is_legit=False,
    )

    print('======= wiki ========')
    dataset_sources['wikipedia'] = process_goggle(
        Path(f'{class_raw}/wikipedia-reliable-sources.goggle'),
        Path(f'{class_proc}/wikipedia.csv'),
    )

    print('======= URL-Phish ========')
    dataset_sources['url-phish'] = process_csv(
        Path(f'{class_raw}/Dataset.csv'),
        Path(f'{class_proc}/url-phish.csv'),
        is_url=True,
        domain_col='url',
        label_col='label',
        inverse=True,
    )

    print('======== Phish&Legit =======')
    dataset_sources['phish-and-legit'] = process_csv(
        Path(f'{class_raw}/new_data_urls.csv'),
        Path(f'{class_proc}/phish-and-legit.csv'),
        is_url=True,
        domain_col='url',
        label_col='status',
        inverse=False,
    )

    print('======== Misinformation domains =========')
    dataset_sources['misinfo-domains'] = process_csv(
        Path(f'{class_raw}/domain_list_clean.csv'),
        Path(f'{class_proc}/misinfo-domains.csv'),
        is_url=False,
        domain_col='url',
        label_col='type',
        inverse=False,
        labels=['unreliable', 'reliable'],
    )

    print('======== URLHaus malware =========')
    dataset_sources['urlhaus'] = process_urlhaus(
        Path(f'{class_raw}/urlhaus.csv'),
        Path(f'{class_proc}/urlhaus.csv'),
    )

    print('======== Merging final labels =========')
    merge_processed_labels(
        class_proc,
        Path(f'{class_proc}/labels.csv'),
        annotated_csv=Path(f'{class_proc}/labels_annot.csv'),
    )

    print('======== Merging with reg scores =========')
    merge_reg_class(
        Path(f'{class_proc}/labels.csv'),
        Path(f'{reg_proc}/domain_pc1.csv'),
        Path(f'{data_dir}/labels.csv'),
    )

    # drop_invalid_domains(Path(f'{data_dir}/labels.csv'))

    print('======== Creating final CSVs =========')
    final_domains = sorted(read_weak_labels(Path(f'{class_proc}/labels.csv')).keys())
    create_merged_annotated(
        final_domains,
        dataset_sources,
        Path(f'{data_dir}/labels_annot.csv'),
        weak_labels=read_weak_labels(Path(f'{class_proc}/labels.csv')),
        reg_scores=read_reg_scores(Path(f'{reg_proc}/domain_pc1.csv')),
    )

    create_labels_csv(
        dataset_sources=dataset_sources,
        reg_scores=read_reg_scores(Path(f'{reg_proc}/domain_pc1.csv')),
        output_csv=Path(f'{data_dir}/labels.csv'),
    )

    path = Path('data/labels.csv')

    total = 0
    non_null = 0

    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('reg_score') not in (None, '', 'NA'):
                non_null += 1

    print(f'Total rows: {total}')
    print(f'Rows with reg_score: {non_null}')


if __name__ == '__main__':
    main()
