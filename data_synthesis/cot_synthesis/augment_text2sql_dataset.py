import argparse
import json
from pathlib import Path


SQL_COMPLEXITY_VI = {
    "Simple": "Đơn giản",
    "Moderate": "Trung bình",
    "Complex": "Phức tạp",
    "Highly Complex": "Rất phức tạp",
}

QUESTION_STYLE_VI = {
    "Colloquial": "Khẩu ngữ",
    "Formal": "Trang trọng",
    "Interrogative": "Nghi vấn",
    "Vague": "Mơ hồ",
    "Descriptive": "Miêu tả",
    "Multi-turn Dialogue": "Hội thoại nhiều lượt",
    "Imperative": "Mệnh lệnh",
    "Concise": "Ngắn gọn",
    "Metaphorical": "Ẩn dụ",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_db_id2schema(tables_json_path: Path, needed_db_ids: set[str]) -> dict[str, str]:
    tables = load_json(tables_json_path)
    db_id2schema: dict[str, str] = {}
    for t in tables:
        db_id = t.get("db_id")
        if db_id in needed_db_ids and "ddls" in t:
            db_id2schema[db_id] = "\n\n".join(t["ddls"])
    return db_id2schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/synthetic_text2sql_dataset.json",
        help="Path to existing synthetic dataset json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/synthetic_text2sql_dataset.json",
        help="Where to write the augmented dataset json.",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default="../database_synthesis/tables.json",
        help="tables.json path for mapping db_id -> CREATE TABLE ddls.",
    )
    opt = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / opt.input).resolve()
    output_path = (base_dir / opt.output).resolve()
    tables_path = (base_dir / opt.tables).resolve()

    dataset = load_json(input_path)
    if not isinstance(dataset, list):
        raise ValueError("Expected dataset to be a list of dicts.")

    needed_db_ids = {item.get("db_id") for item in dataset if isinstance(item, dict) and "db_id" in item}
    needed_db_ids.discard(None)

    if not tables_path.exists():
        raise FileNotFoundError(f"tables.json not found: {tables_path}")

    db_id2schema = build_db_id2schema(tables_path, needed_db_ids)

    missing_schema_ids = []
    for item in dataset:
        if not isinstance(item, dict):
            continue
        db_id = item.get("db_id")

        # Add schema field (HF SynSQL-2.5M exposes it as a single CREATE TABLE string).
        schema = db_id2schema.get(db_id, "")
        item["schema"] = schema
        if db_id not in db_id2schema:
            missing_schema_ids.append(db_id)

        # Translate style/complexity labels to Vietnamese.
        item["sql_complexity"] = SQL_COMPLEXITY_VI.get(item.get("sql_complexity"), item.get("sql_complexity"))
        item["question_style"] = QUESTION_STYLE_VI.get(
            item.get("question_style"), item.get("question_style")
        )

    # De-dup for readability
    missing_schema_unique = sorted(set(missing_schema_ids))
    if missing_schema_unique:
        print(
            f"[WARN] Missing schema for {len(missing_schema_unique)} db_id(s): "
            f"{missing_schema_unique[:5]}{'...' if len(missing_schema_unique) > 5 else ''}"
        )

    dump_json(output_path, dataset)
    print(f"[OK] Wrote augmented dataset: {output_path}")


if __name__ == "__main__":
    main()

