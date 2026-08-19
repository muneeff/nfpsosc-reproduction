import json
from pathlib import Path
from collections import Counter


RESULT_DIR = Path(
    "outputs/v2/ablation_full/results"
)


def main():

    files = list(RESULT_DIR.glob("*.json"))

    total = len(files)

    status_counter = Counter()
    variant_counter = Counter()
    missing_metrics = 0
    errors = []

    for file in files:
        try:
            data = json.loads(
                file.read_text(
                    encoding="utf-8"
                )
            )

            status_counter[data.get("status")] += 1
            variant_counter[data.get("variant")] += 1

            if data.get("metrics") is None:
                missing_metrics += 1

        except Exception as exc:
            errors.append(
                (file.name, str(exc))
            )


    print("=" * 50)
    print("ABLATION INTEGRITY REPORT")
    print("=" * 50)

    print("Total files:", total)

    print("\nStatus:")
    for k, v in status_counter.items():
        print(k, ":", v)

    print("\nVariants:")
    for k, v in sorted(variant_counter.items()):
        print(k, ":", v)

    print("\nMissing metrics:", missing_metrics)

    print("\nRead errors:", len(errors))

    if errors:
        for e in errors[:10]:
            print(e)


if __name__ == "__main__":
    main()