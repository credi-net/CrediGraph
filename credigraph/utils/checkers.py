# Checker helpers,
# Incl. checkers for processed label files and graph files.

import csv
from pathlib import Path

import regex as re

from credigraph.utils.readers import load_domains

# For labels
# ----------

_DOMAIN_RE = re.compile(r'^[a-z0-9][a-z0-9.-]*\.[a-z0-9.-]+$')


def check_overlaps(strong_labels: Path, weak_labels: Path) -> None:
    """Check and print overlaps between strong and weak label datasets.

    Parameters:
        strong_labels : pathlib.Path
            Path to CSV file containing strong labels,
            with columns 'domain', 'pc1' (to be changed to 'label' when we merge with other sources than DQR)
        weak_labels : pathlib.Path
            Path to CSV file containing weak labels,
            with columns 'domain' and 'label', label = 0 for phishing, label = 1 for legitimate

    Returns:
        None
    """
    strong = load_domains(strong_labels)
    weak = load_domains(weak_labels)

    overlap = strong & weak
    union = strong | weak

    print(f'# strong: {len(strong)}')
    print(f'# weak: {len(weak)}')
    print(f'# overlap: {len(overlap)}')
    print(f'# union: {len(union)}')


def check_processed_labels(processed: Path) -> None:
    """Inspect a processed CSV file and print basic statistics.

    Prints:
      - Total number of non-empty data rows
      - The header row (if present)
      - Counts of label 0 and label 1

    Parameters:
        processed : pathlib.Path
            Path to the processed CSV file.

    Returns: None
    """
    processed_count = 0
    label_counts = {0: 0, 1: 0}
    headers = None

    with processed.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)

        for row in reader:
            if not row:
                continue
            processed_count += 1
            label = int(row[1])
            if label in label_counts:
                label_counts[label] += 1

    print('Processed: rows:', processed_count)
    print('Headers:', headers)
    print('Label counts:')
    print('  0:', label_counts[0])
    print('  1:', label_counts[1])


def is_valid_domain(domain: str | None) -> bool:
    if not domain:
        return False

    d = domain.strip().lower()

    if not d or not d.isascii():
        return False

    if any(ord(ch) < 32 for ch in d):
        return False

    if '..' in d or d.startswith('.') or d.endswith('.'):
        return False

    return bool(_DOMAIN_RE.match(d))


def drop_invalid_domains(csv_path: Path) -> int:
    """Remove rows whose domain field is malformed.

    Keeps the original header and rewrites the file in place, omitting rows
    with non-ASCII/control characters or characters outside the domain pattern
    (letters, digits, hyphens, and dots with at least one dot).

    Returns the number of removed rows.
    """
    if not csv_path.exists():
        return 0

    removed = 0
    kept: list[dict[str, str]] = []

    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames or 'domain' not in fieldnames:
            print(f'[WARN] No domain column found in {csv_path}, skipping cleanup.')
            return 0

        for row in reader:
            domain = row.get('domain')
            if is_valid_domain(domain):
                kept.append(row)
            else:
                removed += 1

    if removed == 0:
        print(f'[INFO] No invalid domains found in {csv_path}')
        return 0

    tmp_path = csv_path.with_suffix(csv_path.suffix + '.tmp')

    with tmp_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    tmp_path.replace(csv_path)

    print(f'[INFO] Removed {removed} invalid domain rows from {csv_path}')
    return removed
