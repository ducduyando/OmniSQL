#!/usr/bin/env bash
set -euo pipefail

# Mini Vietnamese SynSQL-like pipeline runner
# Usage:
#   ./run_vi_mini.sh [MAX_TABLES] [MAX_DBS] [SAMPLES_PER_TABLE] [Q_LIMIT] [COT_LIMIT]
#
# Example:
#   ./run_vi_mini.sh 20 20 3 150 120

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create environment first."
  exit 1
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Missing DEEPSEEK_API_KEY. Export your DeepSeek API key first."
  echo 'Example: export DEEPSEEK_API_KEY="sk-xxxx"'
  exit 1
fi

source .venv/bin/activate

MAX_TABLES="${1:-20}"
MAX_DBS="${2:-20}"
SAMPLES_PER_TABLE="${3:-3}"
Q_LIMIT="${4:-150}"
COT_LIMIT="${5:-120}"

MODEL="${MODEL:-deepseek-chat}"
SQL_TEMP="${SQL_TEMP:-0.7}"
Q_TEMP="${Q_TEMP:-0.8}"
COT_TEMP="${COT_TEMP:-0.8}"
Q_SAMPLE_NUM="${Q_SAMPLE_NUM:-3}"
COT_SAMPLE_NUM="${COT_SAMPLE_NUM:-5}"

echo "========== Mini Vietnamese data synthesis =========="
echo "MODEL=$MODEL"
echo "MAX_TABLES=$MAX_TABLES, MAX_DBS=$MAX_DBS, SAMPLES_PER_TABLE=$SAMPLES_PER_TABLE"
echo "Q_LIMIT=$Q_LIMIT, COT_LIMIT=$COT_LIMIT"
echo "Q_SAMPLE_NUM=$Q_SAMPLE_NUM, COT_SAMPLE_NUM=$COT_SAMPLE_NUM"
echo "==================================================="

echo
echo "[1/4] Database synthesis"
cd "$ROOT_DIR/data_synthesis/database_synthesis"
mkdir -p prompts results synthetic_sqlite_databases
python3 generate_schema_synthesis_prompts.py --max_tables "$MAX_TABLES" --table_num_min 4 --table_num_max 6
python3 synthesize_schema.py --model "$MODEL"
python3 generate_schema_enhancement_prompts.py
python3 enhance_schema.py --model "$MODEL"
python3 build_sqlite_databases.py
python3 generate_tables_json.py

echo
echo "[2/4] SQL synthesis"
cd "$ROOT_DIR/data_synthesis/sql_synthesis"
mkdir -p prompts results
python3 generate_sql_synthesis_prompts.py --max_dbs "$MAX_DBS" --samples_per_table "$SAMPLES_PER_TABLE"
python3 synthesize_sql.py --model "$MODEL" --temperature "$SQL_TEMP"
python3 post_process_sqls.py

echo
echo "[3/4] Vietnamese question synthesis"
cd "$ROOT_DIR/data_synthesis/question_synthesis"
mkdir -p prompts results embeddings
python3 generate_question_synthesis_prompts.py --limit "$Q_LIMIT"
python3 synthesize_question.py --model "$MODEL" --sample_num "$Q_SAMPLE_NUM" --temperature "$Q_TEMP"
python3 post_process_questions.py

echo
echo "[4/4] CoT synthesis"
cd "$ROOT_DIR/data_synthesis/cot_synthesis"
mkdir -p prompts results
python3 generate_cot_synthesis_prompts.py --limit "$COT_LIMIT"
python3 synthesize_cot.py --model "$MODEL" --sample_num "$COT_SAMPLE_NUM" --temperature "$COT_TEMP"
python3 post_process_cot.py

echo
echo "Done."
echo "Final dataset:"
echo "  $ROOT_DIR/data_synthesis/cot_synthesis/results/synthetic_text2sql_dataset.json"

