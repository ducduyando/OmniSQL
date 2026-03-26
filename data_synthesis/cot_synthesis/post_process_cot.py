import json
import re
import sqlite3
import os
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys
import ijson
import random
from pathlib import Path

def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

def execute_sql(data_idx, db_file, sql):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res)  # `set` operation on execution results, and make `set` hashable
        conn.rollback()  # Roll back any changes
        return data_idx, db_file, sql, execution_res, 1
    except:
        conn.rollback()  # Ensure rollback on exception
        return data_idx, db_file, sql, None, 0
    finally:
        conn.close()

def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        res = (data_idx, db_file, sql, None, 0)

    return res

def execute_callback_execute_sqls(result):
    execution_results.append(result)

def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1):
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, sql, timeout), callback=execute_callback_execute_sqls)
    pool.close()
    pool.join()

def load_json_file(file):
    dataset = []
    with open(file, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        for obj in objects:
            dataset.append(obj)
    return dataset

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    results_path = base_dir / "results" / "cot_synthesis.json"
    output_path = base_dir / "results" / "synthetic_text2sql_dataset.json"

    results = load_json_file(str(results_path))
    
    if len(results) == 0:
        print("No CoT synthesis results found; writing empty dataset.")
        with open("results/synthetic_text2sql_dataset.json", "w", encoding="utf-8") as f:
            f.write(json.dumps([], ensure_ascii=False, indent=2))
        raise SystemExit(0)

    response_lens = [len(r.get("responses", [])) for r in results]
    if min(response_lens) == 0:
        print("Some CoT samples have 0 responses; truncating to min>0 may not be possible.")
        # Filter out samples with empty responses to keep voting stable.
        results = [r for r in results if len(r.get("responses", [])) > 0]
        response_lens = [len(r.get("responses", [])) for r in results]

    sampling_num = min(response_lens)
    print("sampling_num (min responses across samples):", sampling_num)

    # Ensure each sample has exactly sampling_num responses.
    for r in results:
        r["responses"] = r["responses"][:sampling_num]

    # execution results-guided major voting
    major_voting_filter_num = 0
    major_voting_results = []
    process_batch_size = 10240

    # Build schema map for all db_ids appearing in cot_synthesis.json.
    # `tables.json` stores `ddls` (CREATE TABLE statements) for each `db_id`.
    sql_complexity_map = {
        "Simple": "Đơn giản",
        "Moderate": "Trung bình",
        "Complex": "Phức tạp",
        "Highly Complex": "Rất phức tạp",
    }
    question_style_map = {
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

    needed_db_ids = set(r["db_id"] for r in results if "db_id" in r)
    tables_path = base_dir.parent / "database_synthesis" / "tables.json"
    db_id2schema = {}
    if tables_path.exists():
        tables = json.load(open(str(tables_path), "r", encoding="utf-8"))
        for t in tables:
            db_id = t.get("db_id")
            if db_id in needed_db_ids and "ddls" in t:
                db_id2schema[db_id] = "\n\n".join(t["ddls"])
    else:
        print(f"[WARN] tables.json not found at: {tables_path}")

    missing_schema_db_ids = []

    for pred_idx in tqdm(range(0, len(results), process_batch_size)):
        batch_cot_results = results[pred_idx: pred_idx + process_batch_size]

        batch_db_files = []
        batch_sqls = []
        execution_results = []
        for cot_result in batch_cot_results:
            batch_db_files.extend([os.path.join("../database_synthesis/synthetic_sqlite_databases", cot_result["db_id"], cot_result["db_id"] + ".sqlite")] * sampling_num)
            batch_sqls.extend([parse_response(response) for response in cot_result["responses"]])
        assert len(batch_db_files) == len(batch_sqls)
        execute_sqls_parallel(batch_db_files, batch_sqls, 20, 2)
        execution_results = sorted(execution_results, key = lambda x: x[0])

        assert len(batch_cot_results) * sampling_num == len(execution_results)

        for data_idx in range(len(batch_cot_results)):
            cot_result = batch_cot_results[data_idx]
            execution_results_in_one_sample = execution_results[sampling_num * data_idx: sampling_num * (data_idx + 1)]
            assert len(cot_result["responses"]) == len(execution_results_in_one_sample)

            major_voting_dict = dict()
            for cot, execution_result in zip(cot_result["responses"], execution_results_in_one_sample):
                if execution_result[-1] == 0: # invalid SQL queries
                    continue

                if execution_result[-2] in major_voting_dict:
                    major_voting_dict[execution_result[-2]].append(cot)
                else:
                    major_voting_dict[execution_result[-2]] = [cot]
            
            # if the number of valid cots is less than 3, we discard current data sample
            valid_cot_num = sum([len(cot_list) for cot_list in major_voting_dict.values()])
            # print("valid_cot_num:", valid_cot_num)
            if valid_cot_num < 3:
                major_voting_filter_num += 1
                continue
            
            # find cots with the most vote count, based on the execution results
            voting_key = max(major_voting_dict, key = lambda k: len(major_voting_dict[k]))
            voting_cots = major_voting_dict[voting_key]
            final_cot = random.choice(voting_cots)

            major_voting_results.append(
                {
                    "db_id": cot_result["db_id"],
                    "sql_complexity": sql_complexity_map.get(cot_result["sql_complexity"], cot_result["sql_complexity"]),
                    "question_style": question_style_map.get(cot_result["question_style"], cot_result["question_style"]),
                    "question": cot_result["question"],
                    "external_knowledge": cot_result["external_knowledge"],
                    "cot": final_cot,
                    "sql": parse_response(final_cot),
                    # Required by SynSQL-2.5M format on HF: a single string containing CREATE TABLE statements.
                    "schema": db_id2schema.get(cot_result["db_id"], "")
                }
            )
            if cot_result["db_id"] not in db_id2schema:
                missing_schema_db_ids.append(cot_result["db_id"])
    print("major_voting_filter_num:", major_voting_filter_num)
    print("num of data samples (after execution-based major voting):", len(major_voting_results))
    if missing_schema_db_ids:
        # de-dup for readability
        missing_unique = sorted(set(missing_schema_db_ids))
        print(f"[WARN] Missing schema for {len(missing_unique)} db_id(s): {missing_unique[:5]}{'...' if len(missing_unique) > 5 else ''}")
    
    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write(json.dumps(major_voting_results, ensure_ascii=False, indent=2))