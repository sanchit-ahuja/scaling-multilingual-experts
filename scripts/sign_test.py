#!/usr/bin/env python3
"""Cross-family / cross-language consistency (sign) tests for the rebuttal.

Individual per-family gaps are modest, but the *direction* of key effects repeats
across independent families and languages. We quantify that with a binomial sign
test at both granularities:
  - per family (n=5): the headline framing, but low power (5/5 -> two-sided p=0.0625);
  - per language / per (language,direction) (n up to ~32/64): the stronger test.

Data are the paper's per-language appendix tables (ground truth); no recompute.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from scipy.stats import binomtest

PAPER = Path("/u/sahuja1/6a014ac79b78946b0c4114eb/tables")
FAMILIES = ["Slavic", "Germanic", "Indic", "Austronesian", "Romance"]
STRAT = {"base": 0, "dense": 1, "expert": 2, "dense-reverted": 3,
         "expert-reverted": 4, "freeze": 5, "layer-reg": 6, "expert-soup": 7}

# Main-table family averages (for the per-family test).
BELE_MAIN = {
    "Slavic": [.719, .619, .726, .697, .733, .740, .719, .734],
    "Germanic": [.758, .642, .756, .729, .760, .758, .755, .761],
    "Indic": [.601, .535, .593, .596, .598, .625, .621, .629],
    "Austronesian": [.668, .587, .661, .667, .685, .698, .680, .682],
    "Romance": [.748, .625, .746, .716, .755, .759, .745, .749],
}
FLORES_MAIN = {
    "Slavic": [33.6, 52.7, 47.4, 53.6, 48.6, 45.7, 44.1, 46.2],
    "Germanic": [34.8, 59.0, 57.0, 59.5, 57.1, 58.0, 49.4, 50.4],
    "Indic": [29.9, 43.5, 36.1, 44.2, 40.7, 40.1, 36.2, 42.6],
    "Austronesian": [33.8, 55.7, 25.0, 53.6, 27.8, 35.2, 42.4, 48.1],
    "Romance": [35.0, 58.2, 55.0, 59.0, 48.2, 44.3, 53.1, 49.6],
}


def parse_perlang(fname: str):
    """Yield (family, [8 float values]) rows from a per-language appendix table."""
    path = PAPER / fname
    last_fam, out = None, []
    for line in open(path):
        if "&" not in line or r"\\" not in line:
            continue
        cols = [c.strip() for c in line.split(r"\\")[0].split("&")]
        if len(cols) != 10:
            continue
        fam = cols[0] if cols[0] in FAMILIES else last_fam
        if fam is None:
            continue
        last_fam = fam
        vals, ok = [], True
        for c in cols[2:10]:
            m = re.search(r"(-?\d+\.\d+)", c)
            if not m:
                ok = False
                break
            vals.append(float(m.group(1)))
        if ok:
            out.append((fam, vals))
    return out


def sign_test(units, better: str, worse: str):
    """units: list of (family, values[8]). Returns dict with win/loss/tie + binomial p."""
    bi, wi = STRAT[better], STRAT[worse]
    win = loss = tie = 0
    fam_win = {f: 0 for f in FAMILIES}
    fam_tot = {f: 0 for f in FAMILIES}
    for fam, v in units:
        d = v[bi] - v[wi]
        fam_tot[fam] += 1
        if d > 1e-9:
            win += 1
            fam_win[fam] += 1
        elif d < -1e-9:
            loss += 1
        else:
            tie += 1
    n = win + loss
    p = binomtest(win, n, 0.5, alternative="two-sided").pvalue if n else float("nan")
    p1 = binomtest(win, n, 0.5, alternative="greater").pvalue if n else float("nan")
    return {"win": win, "loss": loss, "tie": tie, "n": n, "p2": p, "p1": p1,
            "fam_win": fam_win, "fam_tot": fam_tot}


def per_family(main: dict, better: str, worse: str):
    bi, wi = STRAT[better], STRAT[worse]
    k = sum(1 for f in FAMILIES if main[f][bi] >= main[f][wi] - 1e-9)
    p1 = binomtest(k, 5, 0.5, alternative="greater").pvalue
    p2 = binomtest(k, 5, 0.5, alternative="two-sided").pvalue
    return k, p1, p2


def main():
    bele = parse_perlang("belebele_perlang.tex")
    flores = parse_perlang("flores_en_xx_perlang.tex") + parse_perlang("flores_xx_en_perlang.tex")

    claims = [
        ("Belebele: Freeze >= Base (comprehension preserved)", bele, BELE_MAIN, "freeze", "base"),
        ("Belebele: Freeze >= Dense (less forgetting)", bele, BELE_MAIN, "freeze", "dense"),
        ("Belebele: Expert-Reverted >= Dense", bele, BELE_MAIN, "expert-reverted", "dense"),
        ("Belebele: Expert-Soup >= Dense", bele, BELE_MAIN, "expert-soup", "dense"),
        ("Belebele: Layer-Reg >= Base", bele, BELE_MAIN, "layer-reg", "base"),
        ("FLORES: Dense >= Expert (experts lag on MT)", flores, FLORES_MAIN, "dense", "expert"),
        ("FLORES: Dense-Reverted >= Dense", flores, FLORES_MAIN, "dense-reverted", "dense"),
        ("FLORES: Dense-Reverted >= Expert", flores, FLORES_MAIN, "dense-reverted", "expert"),
    ]

    lines = []
    hdr = f"{'Claim':<50} {'per-lang (win/n)':>16} {'p(2s)':>8} {'per-fam':>8} {'p1(fam)':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name, units, main, better, worse in claims:
        r = sign_test(units, better, worse)
        k5, p1f, p2f = per_family(main, better, worse)
        unit = "lang" if "Belebele" in name else "lang.dir"
        lines.append(f"{name:<50} {r['win']:>4}/{r['n']:<3}({r['tie']}t) {unit:<5} "
                     f"{r['p2']:>8.1e} {k5}/5{'':<4} {p1f:>8.3f}")
        fw = "   families: " + ", ".join(f"{f[:3]}:{r['fam_win'][f]}/{r['fam_tot'][f]}" for f in FAMILIES)
        lines.append(fw)
    out = "\n".join(lines)
    print(out)
    dest = Path(__file__).resolve().parents[1] / "rebuttal" / "sign_test.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out + "\n")
    print(f"\nWrote {dest}")
    print("\nNote: per-family n=5 has low power (5/5 -> one-sided p=0.031, two-sided p=0.0625);")
    print("the per-language sign test is the stronger evidence of a consistent effect direction.")


if __name__ == "__main__":
    main()
