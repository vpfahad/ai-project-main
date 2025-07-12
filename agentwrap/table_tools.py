from typing import List, Dict, Any

def parse_table(table_str: str) -> List[Dict[str, Any]]:
    """
    Parses a markdown or CSV table string into a list of dicts.
    Assumes first row is header.
    """
    import csv
    from io import StringIO

    # Try to detect and remove markdown table pipes
    lines = [line.strip() for line in table_str.strip().splitlines() if line.strip()]
    if lines and lines[0].startswith("|") and "|" in lines[0]:
        # Remove leading/trailing pipes and split
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in lines
            if "---" not in line  # skip markdown separator
        ]
    else:
        # Assume CSV
        reader = csv.reader(StringIO(table_str))
        rows = [row for row in reader if row]

    if not rows or len(rows) < 2:
        return []

    header = rows[0]
    data_rows = rows[1:]
    table = []
    for row in data_rows:
        if len(row) != len(header):
            continue
        table.append({header[i]: row[i] for i in range(len(header))})
    return table

def get_table_min_max(table: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each numeric column, find min/max values and the row(s) they occur in.
    Returns: {col: {"min": (value, [row_idxs]), "max": (value, [row_idxs])}}
    """
    import re

    if not table:
        return {}

    header = list(table[0].keys())
    col_values = {col: [] for col in header}
    for row in table:
        for col in header:
            val = row[col]
            # Try to convert to float if possible
            try:
                num = float(re.sub(r"[^\d\.\-eE]", "", val))
                col_values[col].append(num)
            except Exception:
                col_values[col].append(None)

    stats = {}
    for col, values in col_values.items():
        # Only consider columns with at least one numeric value
        numeric_vals = [(i, v) for i, v in enumerate(values) if v is not None]
        if not numeric_vals:
            continue
        idxs, nums = zip(*numeric_vals)
        min_val = min(nums)
        max_val = max(nums)
        min_idxs = [i for i, v in zip(idxs, nums) if v == min_val]
        max_idxs = [i for i, v in zip(idxs, nums) if v == max_val]
        stats[col] = {
            "min": (min_val, min_idxs),
            "max": (max_val, max_idxs),
        }
    return stats