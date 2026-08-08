import json
import sys
from pathlib import Path

import summarization as s


def main():
    model_key, start, end, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    model_name = s.MODELS[model_key]

    dataset = s.load_dataset("cnn_dailymail/validation.csv")
    chunk = dataset[start:end]

    results = s.evaluate_model(chunk, model_name, limit=len(chunk))

    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} results for {model_key} [{start}:{end}] -> {out_path}")


if __name__ == "__main__":
    main()
