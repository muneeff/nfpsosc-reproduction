from pathlib import Path
import json


def ablation_result_path(
    workspace,
    run_id,
):
    return (
        Path(workspace)
        / "results"
        / f"{run_id}.json"
    )


def ensure_ablation_workspace(workspace):

    root = Path(workspace)

    (root / "results").mkdir(
        parents=True,
        exist_ok=True,
    )

    return root


def save_ablation_result(
    workspace,
    run_id,
    payload,
):

    path = ablation_result_path(
        workspace,
        run_id,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def load_existing_results(workspace):

    root = Path(workspace) / "results"

    if not root.exists():
        return {}

    results = {}

    for file in root.glob("*.json"):
        results[file.stem] = json.loads(
            file.read_text(
                encoding="utf-8"
            )
        )

    return results


__all__ = [
    "ensure_ablation_workspace",
    "save_ablation_result",
    "load_existing_results",
]