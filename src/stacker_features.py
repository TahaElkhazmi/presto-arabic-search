"""Deterministic text features for the tabular stacker (see TECHNICAL_REPORT.md section 6).

Two feature families that the neural signals miss:
  - multi-scale character n-grams (n=2,3,4): Jaccard + containment, for typo /
    dialect / transliteration / model-number-prefix robustness on short titles.
  - numeric + physical-unit parsing: num_match_count, num_conflict, unit_conflict,
    with model-number exclusion so "s23" vs "s24" does not fire a false penalty.

All text is run through common.normalize_arabic first (Arabic-Indic digits ->
western, diacritics stripped, alef/yeh/heh unified, lowercased). N-gram sets are
meant to be cached per UNIQUE text (there are ~82k queries and ~107k names, not
millions of pairs) - see build_ngram_cache.
"""

import re

from common import normalize_arabic

# ---- character n-grams -------------------------------------------------------

NGRAM_NS = (2, 3, 4)


def char_ngrams(norm_text: str, n: int) -> frozenset[str]:
    s = norm_text.strip()
    if len(s) < n:
        return frozenset({s}) if s else frozenset()
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


def build_ngram_cache(norm_texts) -> dict[str, dict[int, frozenset[str]]]:
    """text -> {n: ngram set} for each unique normalized text."""
    cache: dict[str, dict[int, frozenset[str]]] = {}
    for t in set(norm_texts):
        cache[t] = {n: char_ngrams(t, n) for n in NGRAM_NS}
    return cache


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def containment(a: frozenset, b: frozenset) -> float:
    """Share of a's grams present in b (query-in-name when a=query)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ---- numeric + unit parsing --------------------------------------------------

# canonical unit -> dimension class; Arabic forms are post-normalize_arabic
# (teh-marbuta->heh: بوصة->بوصه; alef variants already folded).
_UNIT_CLASS = {
    # length
    "m": "len", "cm": "len", "mm": "len", "inch": "len", "in": "len",
    "متر": "len", "سم": "len", "مم": "len", "انش": "len", "بوصه": "len",
    # volume
    "ml": "vol", "l": "vol", "لتر": "vol", "مل": "vol",
    # weight
    "kg": "wt", "g": "wt", "mg": "wt", "كجم": "wt", "كيلو": "wt",
    "جرام": "wt", "جم": "wt", "غرام": "wt",
    # storage
    "gb": "stor", "tb": "stor", "mb": "stor", "kb": "stor",
    "غيغا": "stor", "جيجا": "stor", "تيرا": "stor", "ميغا": "stor",
    # battery capacity
    "mah": "batt", "امبير": "batt",
    # voltage / power / frequency
    "v": "volt", "فولت": "volt", "فولط": "volt",
    "w": "pow", "kw": "pow", "واط": "pow", "وات": "pow",
    "hz": "freq", "khz": "freq", "mhz": "freq", "ghz": "freq", "هرتز": "freq",
}
# longest-first alternation so "mm" wins over "m", "بوصه" over ...
_UNITS_SORTED = sorted(_UNIT_CLASS, key=len, reverse=True)
_UNIT_ALT = "|".join(re.escape(u) for u in _UNITS_SORTED)
# a number (int/decimal) followed by optional space and a known unit
_NUM_UNIT = re.compile(rf"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*({_UNIT_ALT})(?![a-zء-ي])")
# a standalone bare number: digits not glued to any letter on either side
_BARE_NUM = re.compile(r"(?<![a-z0-9ء-ي])(\d+(?:\.\d+)?)(?![a-z0-9ء-ي])")


def parse_numeric(norm_text: str) -> tuple[set[float], dict[str, set[float]]]:
    """Return (bare_numbers, {unit_class: {values}}).

    Numbers glued to non-unit letters (model codes like s23, t800, a50) are
    excluded from bare_numbers by the surrounding-letter lookarounds; numbers
    directly bearing a known unit are captured with their dimension class.
    """
    units: dict[str, set[float]] = {}
    consumed = []
    for m in _NUM_UNIT.finditer(norm_text):
        val = float(m.group(1))
        cls = _UNIT_CLASS[m.group(2)]
        units.setdefault(cls, set()).add(val)
        consumed.append((m.start(), m.end()))
    # bare numbers, excluding spans already consumed by a unit match
    bare: set[float] = set()
    for m in _BARE_NUM.finditer(norm_text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        bare.add(float(m.group(1)))
    return bare, units


def numeric_features(qnum, nnum) -> tuple[int, int, int]:
    """(num_match_count, num_conflict, unit_conflict) between query and name."""
    q_bare, q_units = qnum
    n_bare, n_units = nnum
    q_all = q_bare | {v for vs in q_units.values() for v in vs}
    n_all = n_bare | {v for vs in n_units.values() for v in vs}
    num_match_count = len(q_all & n_all)
    # both sides carry numbers but share none -> contradiction
    num_conflict = int(bool(q_all) and bool(n_all) and not (q_all & n_all))
    # same dimension present on both sides with disjoint values -> strong conflict
    unit_conflict = 0
    for cls, qv in q_units.items():
        nv = n_units.get(cls)
        if nv and not (qv & nv):
            unit_conflict = 1
            break
    return num_match_count, num_conflict, unit_conflict


def build_numeric_cache(norm_texts) -> dict[str, tuple[set[float], dict[str, set[float]]]]:
    return {t: parse_numeric(t) for t in set(norm_texts)}
