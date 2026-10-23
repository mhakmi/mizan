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


def word_count_as_tokens(text):
    return len(text.split())


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
                "prompt_tokens": word_count_as_tokens(text),
            }
        print(json.dumps(line))


if __name__ == "__main__":
    main()
