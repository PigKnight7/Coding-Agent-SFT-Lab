"""Partition original assertions; never synthesize expected answers or expose private fixtures."""
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestPartition:
    public: str
    hidden: str
    public_count: int
    hidden_count: int


def partition_tests(repo):
    paths = sorted((Path(repo) / 'tests').glob('test_*.py'))
    if len(paths) != 1:
        raise ValueError('Exactly one original benchmark test file is required')
    source = paths[0].read_text()
    tree = ast.parse(source)
    assertions = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    # Restrict to independent direct assertions. Shared loops/fixtures can leak held-out
    # inputs or depend on assertions with side effects; exclude instead of guessing.
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    direct = [n for f in functions for n in f.body if isinstance(n, ast.Assert)]
    if len(assertions) < 2 or len(direct) != len(assertions):
        raise ValueError('Need at least two independent direct original assertions')
    for f in functions:
        if any(isinstance(n, ast.Assert) for n in f.body) and any(
            not isinstance(n, (ast.Assert, ast.Pass)) and not (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            ) for n in f.body
        ):
            raise ValueError('Shared test setup cannot be safely partitioned')
    if any(isinstance(n, (ast.NamedExpr, ast.Global, ast.Nonlocal)) for a in assertions for n in ast.walk(a)):
        raise ValueError('Assertions with side effects cannot be partitioned')
    ordered = sorted(assertions, key=lambda n: n.lineno)
    unique = list(dict.fromkeys(ast.dump(n) for n in ordered))
    if len(unique) < 2:
        raise ValueError("Repeated copies of one assertion cannot form hidden cases")
    hidden_keys = set(unique[1::2])
    hidden_lines = {n.lineno for n in ordered if ast.dump(n) in hidden_keys}

    class Filter(ast.NodeTransformer):
        def __init__(self, hidden):
            self.hidden = hidden

        def visit_Assert(self, node):
            return node if (node.lineno in hidden_lines) == self.hidden else ast.copy_location(ast.Pass(), node)

    def render(hidden):
        return ast.unparse(ast.fix_missing_locations(Filter(hidden).visit(ast.parse(source)))) + '\n'

    return TestPartition(render(False), render(True), len(ordered) - len(hidden_lines), len(hidden_lines))
