"""Measure MDP depth of every program in src/data/rule/functions.txt."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lang.parser import parse
from src.lang.ast_nodes import (
    NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode,
)


def mdp_depth(node) -> int:
    """
    Depth as consumed by SynthesisState.depth_budget.

    LAMBDA / APPLY / IF each cost 1 on the longest path.
    Terminals (numbers, bools, variables, empty list) cost 0.
    Non-empty list literals are not directly emittable by the RL action
    vocabulary, so we treat them as 1 + max(child depths) — they would
    require a cons-chain of equivalent depth to reproduce.
    """
    if isinstance(node, (NumberNode, BooleanNode, VariableNode)):
        return 0
    if isinstance(node, ListNode):
        if not node.elements:
            return 0
        return 1 + max(mdp_depth(e) for e in node.elements)
    if isinstance(node, LambdaNode):
        return 1 + mdp_depth(node.body)
    if isinstance(node, IfNode):
        return 1 + max(
            mdp_depth(node.condition),
            mdp_depth(node.then_expr),
            mdp_depth(node.else_expr),
        )
    if isinstance(node, ApplicationNode):
        child_depths = [mdp_depth(node.function)] + [mdp_depth(a) for a in node.arguments]
        return 1 + max(child_depths)
    raise ValueError(f"Unknown node type: {type(node)}")


def has_nonempty_list_literal(node) -> bool:
    if isinstance(node, ListNode) and node.elements:
        return True
    if isinstance(node, LambdaNode):
        return has_nonempty_list_literal(node.body)
    if isinstance(node, ApplicationNode):
        return has_nonempty_list_literal(node.function) or any(
            has_nonempty_list_literal(a) for a in node.arguments
        )
    if isinstance(node, IfNode):
        return (has_nonempty_list_literal(node.condition)
                or has_nonempty_list_literal(node.then_expr)
                or has_nonempty_list_literal(node.else_expr))
    if isinstance(node, ListNode):
        return any(has_nonempty_list_literal(e) for e in node.elements)
    return False


def main():
    path = Path(__file__).resolve().parents[1] / "src" / "data" / "rule" / "functions.txt"
    lines = path.read_text().splitlines()

    depths = []
    failed = []
    per_line = []

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            ast = parse(line)
            d = mdp_depth(ast)
            depths.append(d)
            per_line.append((i, d, line, has_nonempty_list_literal(ast)))
        except Exception as e:
            failed.append((i, line, str(e)))

    print(f"Parsed: {len(depths)} programs")
    print(f"Failed: {len(failed)}")
    if failed:
        for i, line, err in failed[:5]:
            print(f"  line {i}: {err}  :: {line[:80]}")

    print()
    print("Depth distribution:")
    ctr = Counter(depths)
    for d in sorted(ctr):
        bar = "#" * ctr[d]
        print(f"  depth {d:2d}: {ctr[d]:4d}  {bar}")

    print()
    print(f"Min depth: {min(depths)}")
    print(f"Max depth: {max(depths)}")
    print(f"Mean:      {sum(depths)/len(depths):.2f}")
    print(f"Median:    {sorted(depths)[len(depths)//2]}")

    print()
    print("Deepest 10 programs:")
    per_line.sort(key=lambda t: t[1], reverse=True)
    for i, d, line, has_lit in per_line[:10]:
        tag = " [uses non-empty list literal]" if has_lit else ""
        short = line if len(line) <= 110 else line[:107] + "..."
        print(f"  line {i:3d}  depth={d}{tag}")
        print(f"    {short}")

    print()
    for cap in (10, 11, 12, 13, 14, 16):
        n_ok = sum(1 for d in depths if d <= cap)
        pct = 100 * n_ok / len(depths)
        print(f"  max_depth={cap:2d}: reproduces {n_ok}/{len(depths)} ({pct:.1f}%)")

    n_lit = sum(1 for _, _, _, has_lit in per_line if has_lit)
    print()
    print(f"Programs using non-empty list literals (not in RL action vocab): {n_lit}")


if __name__ == "__main__":
    main()
