"""Read the receipts in a folder and print one JSON line for each.

This is the program the example scores. It stands in for a real receipt
reader, such as one that sends each scan to a model, and uses only the
standard library.
"""

import json
import sys
from pathlib import Path


def read_receipt(text):
    values = {"shop": None, "date": None, "total": None, "tip": None}
    for line in text.splitlines():
        label, _, value = line.partition(":")
        label = label.strip().lower()
        value = value.strip()
        if label == "shop":
            values["shop"] = value
        elif label == "date":
            values["date"] = value
        elif label == "total":
            values["total"] = read_money(value)
        elif label == "tip":
            values["tip"] = read_money(value)
    return values


def read_money(text):
    amount, _, currency = text.partition(" ")
    return {"amount": amount.replace(",", ""), "currency": currency or None}


def main():
    folder = Path(sys.argv[1])
    for path in sorted(folder.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            line = {"id": path.stem, "output": None, "error": "not UTF-8 text", "prompt_tokens": 0}
        else:
            line = {
                "id": path.stem,
                "output": read_receipt(text),
                "error": None,
                # A real reader would report the tokens its model call used.
                "prompt_tokens": len(text.split()),
            }
        print(json.dumps(line))


if __name__ == "__main__":
    main()
