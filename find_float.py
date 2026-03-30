#!/usr/bin/env python3
"""
float-gen: Find parsimonious arithmetic expressions that produce a given float.

Given a "surprising" floating-point value, searches for the simplest sequence
of elementary arithmetic operations on common literal values that could have
produced it.  After finding all equally-simple expressions it explores the
template — sweeping each distinct literal over a wide range to reveal which
values produce the same result and which do not.

Algorithm: bottom-up dynamic programming keyed on literal count.
  - Level 1: atomic literals (0.1, 0.2, ..., 1, 2, ..., 20, ...)
  - Level k: all (level-i op level-(k-i)) combinations for 1 ≤ i < k
  - Float values are deduped by exact IEEE-754 bit pattern.
  - The entire matching level is scanned so ALL equally-parsimonious
    expressions are returned, not just the first one found.
  - Commutative pairs (a+b / b+a, a*b / b*a) are normalised to one form.
  - After a match, each distinct literal is swept independently over a large
    candidate set; same-valued literals are replaced together (they represent
    the same "variable" in the template).

Usage:
    python float_gen.py                              # default demo target
    python float_gen.py 0.30000000000000004
    python float_gen.py 0.05000000000001137 -m 6
    python float_gen.py 0.6000000000000001 -t 120
    python float_gen.py 0.30000000000000004 --no-explore
    python float_gen.py 0.05000000000001137 -q
"""

import sys
import struct
import math
import time
import argparse
from decimal import Decimal


# ── Float utilities ────────────────────────────────────────────────────────────

def float_bits(f: float) -> int:
    """Return the IEEE-754 bit pattern of f as a uint64."""
    (b,) = struct.unpack("Q", struct.pack("d", f))
    return b


def ulp_dist(a: float, b: float) -> int:
    """Absolute ULP distance between two finite floats."""
    return abs(float_bits(a) - float_bits(b))


# ── Expression type ────────────────────────────────────────────────────────────

class Expr:
    """
    An arithmetic expression together with its exact float value.
    Leaf nodes (atoms) have left=right=op=None.
    Internal nodes store child expressions and the operator symbol.
    """
    __slots__ = ("value", "text", "n_lits", "left", "op", "right")

    def __init__(
        self,
        value: float,
        text: str,
        n_lits: int,
        left: "Expr | None" = None,
        op: "str | None" = None,
        right: "Expr | None" = None,
    ) -> None:
        self.value = value
        self.text = text
        self.n_lits = n_lits
        self.left = left    # None for atoms
        self.op = op        # '+' | '-' | '*' | '/' | None
        self.right = right  # None for atoms


# ── Atoms (level-1 expressions) ───────────────────────────────────────────────

def build_atoms() -> dict[int, Expr]:
    """Return a dict (bits → Expr) of common, simple float literals."""
    pool: dict[int, Expr] = {}

    def add(v: float, label: str) -> None:
        if not math.isfinite(v):
            return
        b = float_bits(v)
        if b not in pool:
            pool[b] = Expr(v, label, 1)

    # Small positive integers — the most natural operands
    for n in range(1, 21):
        add(float(n), str(n))

    # A few larger round numbers people commonly use
    for n in (25, 50, 100, 200, 500, 1000):
        add(float(n), str(n))

    # Tenths: the most common source of float surprises
    for d in range(1, 10):
        add(d / 10.0, f"0.{d}")

    # Hundredths
    for d in range(1, 10):
        add(d / 100.0, f"0.0{d}")

    # Thousandths
    for d in range(1, 10):
        add(d / 1000.0, f"0.00{d}")

    # Negative powers of 2 (exact in IEEE-754, useful for scaling)
    for k in range(1, 16):
        add(2.0 ** (-k), f"1/{2**k}")

    # Negative powers of 10
    for k in range(1, 8):
        add(10.0 ** (-k), f"1e-{k}")

    # Large powers of 2 and 10 (useful for reciprocals)
    for k in range(5, 11):
        add(2.0 ** k, f"2^{k}")
    for k in range(2, 7):
        add(10.0 ** k, f"10^{k}")

    return pool


# ── DP combination ─────────────────────────────────────────────────────────────

def _expand(
    ea: Expr,
    eb: Expr,
    target_bits: int,
    known: dict[int, Expr],
    staging: dict[int, Expr],
    hits: list[Expr],
    seen_texts: set[str],
) -> None:
    """
    Apply all four binary ops to (ea, eb).

    - Results equal to the target are appended to *hits* (deduplicated by
      normalised text: commutative ops are sorted so a+b and b+a count once).
    - Other previously-unseen results are deposited into *staging*.
    """
    av, bv = ea.value, eb.value
    n = ea.n_lits + eb.n_lits

    candidates: list[tuple[float, str, str, Expr, Expr]] = [
        (av + bv, '+', ea, eb),
        (av - bv, '-', ea, eb),
        (av * bv, '*', ea, eb),
    ]
    if bv != 0.0:
        candidates.append((av / bv, '/', ea, eb))

    for rv, op, left, right in candidates:
        if not math.isfinite(rv):
            continue

        # Normalise commutative ops to avoid (a+b) and (b+a) as duplicates
        if op in ('+', '*') and left.text > right.text:
            left, right = right, left

        text = f"({left.text} {op} {right.text})"
        bits = float_bits(rv)

        if bits == target_bits:
            if text not in seen_texts:
                seen_texts.add(text)
                hits.append(Expr(rv, text, n, left=left, op=op, right=right))
        elif bits not in known and bits not in staging:
            staging[bits] = Expr(rv, text, n, left=ea, op=op, right=eb)


# ── Main search ────────────────────────────────────────────────────────────────

def find_expressions(
    target: float,
    max_lits: int = 7,
    time_limit: float = 60.0,
    verbose: bool = True,
) -> list[Expr]:
    """
    Bottom-up DP search over arithmetic expression complexity.

    Explores expressions with 1, 2, 3, … *max_lits* numeric literals in order.
    Once the minimum literal count k that reaches *target* is found, the entire
    level k is scanned so that ALL equally-parsimonious expressions are returned.
    Returns an empty list if the target is not reached within the given limits.
    """
    target_bits = float_bits(target)

    known: dict[int, Expr] = build_atoms()
    if target_bits in known:
        return [known[target_bits]]

    by_level: dict[int, list[Expr]] = {}
    for e in known.values():
        by_level.setdefault(e.n_lits, []).append(e)

    best_near: "Expr | None" = None
    best_ulps: int = 10 ** 18
    t0 = time.perf_counter()

    for k in range(2, max_lits + 1):
        if time.perf_counter() - t0 >= time_limit:
            if verbose:
                print(f"\n  Time limit reached before level {k}.")
            break

        staging: dict[int, Expr] = {}
        hits: list[Expr] = []
        seen_texts: set[str] = set()
        total_known = sum(len(v) for v in by_level.values())

        if verbose:
            print(f"  Level {k:2d}  ({total_known:>8,} values reachable) ...", end="", flush=True)

        for i in range(1, k):
            j = k - i
            if i not in by_level or j not in by_level:
                continue
            for ea in by_level[i]:
                if time.perf_counter() - t0 >= time_limit:
                    break
                for eb in by_level[j]:
                    _expand(ea, eb, target_bits, known, staging, hits, seen_texts)

        if hits:
            if verbose:
                print(f"  ✓ found {len(hits)} expression(s)!")
            return hits

        for e in staging.values():
            u = ulp_dist(e.value, target)
            if u < best_ulps:
                best_ulps = u
                best_near = e

        added = 0
        for bits, e in staging.items():
            if bits not in known:
                known[bits] = e
                by_level.setdefault(k, []).append(e)
                added += 1

        elapsed = time.perf_counter() - t0
        if verbose:
            line = f"  +{added:>8,} new   ({elapsed:.1f}s elapsed)"
            if best_near is not None and best_ulps <= 20:
                line += f"   [nearest: {best_near.text}  —  {best_ulps} ULP(s) away]"
            print(line, flush=True)

        if added == 0:
            if verbose:
                print("  Search space exhausted — no further float values reachable.")
            break

    if verbose and best_near is not None:
        print(f"\n  Closest expression found ({best_ulps} ULPs from target):")
        print(f"    {best_near.text}  =  {best_near.value!r}")

    return []


# ── Template exploration ───────────────────────────────────────────────────────

def collect_leaves(expr: Expr) -> list[Expr]:
    """Return all leaf (atom) nodes in left-to-right order."""
    if expr.left is None:
        return [expr]
    return collect_leaves(expr.left) + collect_leaves(expr.right)


def eval_tree(expr: Expr, sub: dict[int, float]) -> float:
    """
    Evaluate the expression tree, replacing each leaf whose bits key appears
    in *sub* with the corresponding new value.
    """
    if expr.left is None:
        return sub.get(float_bits(expr.value), expr.value)
    lv = eval_tree(expr.left, sub)
    rv = eval_tree(expr.right, sub)
    if expr.op == '+':  return lv + rv
    if expr.op == '-':  return lv - rv
    if expr.op == '*':  return lv * rv
    if expr.op == '/':  return lv / rv if rv != 0.0 else float('nan')
    raise ValueError(f"Unknown op: {expr.op!r}")


def _sweep_candidates() -> list[float]:
    """Build the candidate pool used when sweeping a single literal."""
    seen: set[int] = set()
    cands: list[float] = []

    def add(v: float) -> None:
        if not math.isfinite(v) or v == 0.0:
            return
        b = float_bits(v)
        if b not in seen:
            seen.add(b)
            cands.append(v)

    for n in range(1, 10_001):
        add(float(n))
    for k in range(31):
        add(2.0 ** k)
        add(-(2.0 ** k))
    for k in range(-10, 11):
        add(10.0 ** k)
    for d in range(1, 100):
        add(d / 10.0)
        add(d / 100.0)
        add(d / 1000.0)
    for n in range(1, 1001):
        add(-float(n))

    return cands


_SWEEP_CANDIDATES: "list[float] | None" = None  # lazily built


def make_template_text(expr: Expr, param_names: dict[int, str]) -> str:
    """Render the expression with parameter names substituted for literals."""
    if expr.left is None:
        return param_names.get(float_bits(expr.value), expr.text)
    left_s  = make_template_text(expr.left,  param_names)
    right_s = make_template_text(expr.right, param_names)
    return f"({left_s} {expr.op} {right_s})"


def explore_template(result: Expr, target: float) -> None:
    """
    For each distinct literal value in *result*, sweep it over a large
    candidate set (keeping all other literals fixed) and report which
    values still produce *target*.  Same-valued literals are replaced
    together — they are the same "variable" in the template.
    """
    global _SWEEP_CANDIDATES
    if _SWEEP_CANDIDATES is None:
        _SWEEP_CANDIDATES = _sweep_candidates()

    target_bits = float_bits(target)
    leaves = collect_leaves(result)

    seen_bits: set[int] = set()
    distinct: list[Expr] = []
    for leaf in leaves:
        b = float_bits(leaf.value)
        if b not in seen_bits:
            seen_bits.add(b)
            distinct.append(leaf)

    param_names: dict[int, str] = {
        float_bits(leaf.value): f"N{idx}"
        for idx, leaf in enumerate(distinct)
    }
    template_str = make_template_text(result, param_names)

    print()
    print("  ── Template exploration " + "─" * 37)
    print(f"  Template : {template_str}")
    for idx, leaf in enumerate(distinct):
        print(f"    N{idx} = {leaf.value!r}  ({leaf.text})")
    print()

    for vary_idx, vary_leaf in enumerate(distinct):
        vary_bits = float_bits(vary_leaf.value)
        label = f"N{vary_idx}"
        fixed_desc = ", ".join(
            f"N{i}={d.value!r}"
            for i, d in enumerate(distinct)
            if float_bits(d.value) != vary_bits
        )
        header = f"  Varying {label} (= {vary_leaf.value!r})"
        if fixed_desc:
            header += f"   [fixed: {fixed_desc}]"
        print(header)

        hits: list[float] = []
        for cand in _SWEEP_CANDIDATES:
            try:
                rv = eval_tree(result, {vary_bits: cand})
            except Exception:
                continue
            if math.isfinite(rv) and float_bits(rv) == target_bits:
                hits.append(cand)

        if not hits:
            print(f"    No other values produce the target.")
        else:
            pos = sorted(v for v in hits if v > 0)
            neg = sorted(v for v in hits if v < 0)

            def fmt_list(vals: list[float], limit: int = 30) -> str:
                strs = [repr(v) for v in vals[:limit]]
                if len(vals) > limit:
                    strs.append(f"… ({len(vals) - limit} more)")
                return "  ".join(strs)

            if pos:
                print(f"    Positive values that work ({len(pos)}):")
                print(f"      {fmt_list(pos)}")
            if neg:
                print(f"    Negative values that work ({len(neg)}):")
                print(f"      {fmt_list(neg)}")

        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find simple arithmetic expressions that produce a given float.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python float_gen.py 0.30000000000000004\n"
               "  python float_gen.py 0.05000000000001137 -m 6\n"
               "  python float_gen.py 0.6000000000000001 -t 120\n"
               "  python float_gen.py 0.30000000000000004 --no-explore\n"
               "  python float_gen.py 0.05000000000001137 -q\n",
    )
    ap.add_argument(
        "target",
        nargs="?",
        type=float,
        default=0.05000000000001137,
        help="Float value to explain  (default: %(default)s)",
    )
    ap.add_argument(
        "-m", "--max-literals",
        type=int,
        default=7,
        metavar="N",
        help="Max number of numeric literals in the expression  (default: %(default)s)",
    )
    ap.add_argument(
        "-t", "--time-limit",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Wall-clock search time limit in seconds  (default: %(default)s)",
    )
    ap.add_argument(
        "--no-explore",
        action="store_true",
        help="Skip template exploration after finding the expression",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = ap.parse_args()

    target: float = args.target
    exact_str = format(Decimal(target), "f")

    try:
        rounded = float(f"{target:.2g}")
        diff = target - rounded
    except Exception:
        rounded = None
        diff = 0.0

    print()
    print(f"  Target      : {target!r}")
    print(f"  Exact value : {exact_str}")
    if rounded is not None and rounded != target:
        print(f"  'Expected'  : {rounded!r}  (off by {diff:+.4e})")
    print()
    print(f"  Searching: up to {args.max_literals} literals, time limit {args.time_limit:.0f}s")
    print()

    results = find_expressions(
        target,
        max_lits=args.max_literals,
        time_limit=args.time_limit,
        verbose=not args.quiet,
    )

    print()
    if results:
        print(f"  ┌─ Result{'s' if len(results) > 1 else ''} ({len(results)} expression(s) at minimum complexity) " + "─" * 20)
        for expr in results:
            print(f"  │  {expr.text}")
        print(f"  │")
        print(f"  │  Value    : {results[0].value!r}")
        print(f"  │  Literals : {results[0].n_lits}  (operations: {results[0].n_lits - 1})")
        print("  └" + "─" * 60)
        if rounded is not None and rounded != target:
            print()
            print(f"  Floating-point rounding accumulates through these operations,")
            print(f"  giving {results[0].value!r} instead of {rounded!r}.")

        if not args.no_explore:
            explore_template(results[0], target)
    else:
        print("  No exact match found within the given limits.")
        print(f"  Suggestions: -m {args.max_literals + 2}  or  -t {int(args.time_limit * 2)}")
    print()


if __name__ == "__main__":
    main()
