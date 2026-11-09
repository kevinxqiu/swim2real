import ast
import json
from pathlib import Path

from swim2real.analysis.numbers import describe_key


def test_every_recorded_number_has_a_complete_definition():
    source = Path(__file__).parents[1] / "records/paper_numbers.json"
    for key, row in json.loads(source.read_text())["values"].items():
        definition, location = describe_key(key)
        assert definition == row["definition"]
        assert location == row["paper_location"]
        assert definition.endswith(".") and "\n" not in definition
        assert definition != key.replace("_", " ").replace(".", ": ")


def test_replay_and_primary_decision_definitions_state_the_rules():
    replay = describe_key("backtracking.replay.beta_0.25.K_2.calls")[0]
    assert "beta=0.25" in replay and "K=2" in replay
    assert "stopping at the first valid strict incumbent improvement" in replay
    primary = describe_key("replication.primary.random.passes")[0]
    assert "Holm-adjusted p-value is below 0.05" in primary
    assert "median paired difference is strictly negative" in primary
    assert "1270 at 1000 mm" in primary and "excluding infeasible seed 1279" in primary


def test_public_analysis_functions_have_type_hints():
    directory = Path(__file__).parents[1] / "src/swim2real/analysis"
    for source in directory.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            assert node.returns is not None, f"{source.name}:{node.name} lacks a return hint"
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                assert argument.annotation is not None, (
                    f"{source.name}:{node.name}.{argument.arg} lacks a hint"
                )
