import json
import os
import random
import sqlite3
import numpy as np
import argparse

from tqdm import tqdm

sql_func_template = '''
### SQL Functions
Bạn có thể cân nhắc một hoặc nhiều SQL function sau khi tạo câu truy vấn:
{sql_funcs}
Lưu ý quan trọng:
Ngoài các function liệt kê ở trên, bạn có thể dùng các function khác miễn là tuân theo syntax của hệ quản trị cơ sở dữ liệu.
'''

insert_stmts_template = '''
### INSERT INTO Statements
Bên dưới là một số câu lệnh `INSERT INTO`. Hãy dùng chúng để hỗ trợ tạo điều kiện lọc (tức mệnh đề `WHERE`) trong câu SQL của bạn:

{insert_statements}
'''

simple_criterion = '''**Tiêu chí:**
SQL mức Simple có thể thỏa một hoặc nhiều điều kiện sau:
- Truy vấn chỉ lấy dữ liệu từ một bảng duy nhất.
- Có thể dùng các hàm tổng hợp cơ bản như `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- Không dùng JOIN; truy vấn phải hoạt động trên một bảng.

**Ví dụ SQL mức Simple:**
```sql
SELECT name, department_name
FROM employees
WHERE level > 5
ORDER BY age DESC;
```'''

moderate_criterion = '''**Tiêu chí:**
SQL mức Moderate có thể thỏa một hoặc nhiều điều kiện sau:
- Có JOIN giữa các bảng, như `JOIN`, `INNER JOIN`, `LEFT JOIN`, `CROSS JOIN`, v.v.
- Có subquery trong mệnh đề `SELECT` hoặc `WHERE`.
- Dùng hàm tổng hợp cùng với mệnh đề `GROUP BY`.
- Có điều kiện `WHERE` phức tạp, gồm `IN`, `BETWEEN`, `LIKE`.
- Có mệnh đề `HAVING` để lọc kết quả tổng hợp.
- Dùng các hàm tổng hợp như `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, v.v.

**Ví dụ SQL mức Moderate:**
```sql
SELECT e.name, d.department_name, AVG(s.salary) AS average_salary
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
LEFT JOIN salaries s ON e.employee_id = s.employee_id
WHERE e.age > 30 AND e.status = 'active'
GROUP BY e.name, d.department_name
HAVING AVG(s.salary) > 50000;
```'''

complex_criterion = '''**Tiêu chí:**
SQL mức Complex có thể thỏa một hoặc nhiều điều kiện sau:
- Có nested subquery phức tạp.
- Dùng nhiều loại JOIN, bao gồm self-join.
- Có window function như `ROW_NUMBER`, `RANK`, v.v.
- Dùng Common Table Expressions (CTE) để tăng tính dễ đọc.
- Kết hợp nhiều hàm tổng hợp.
- Có mệnh đề `WHERE` và `HAVING` phức tạp với nhiều điều kiện.
- Dùng function và toán tử nâng cao.

**Ví dụ SQL mức Complex:**
```sql
WITH EmployeeCTE AS (
    SELECT employee_id, name, department_id, ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT e.name, d.department_name
FROM EmployeeCTE e
INNER JOIN departments d ON e.department_id = d.department_id
WHERE e.rank <= 3;
```'''

highly_complex_criterion = '''**Tiêu chí:**
SQL mức Highly Complex có thể thỏa một hoặc nhiều điều kiện sau:
- Có nhiều CTE để tăng tính dễ đọc.
- Kết hợp nested subquery và nhiều kiểu JOIN khác nhau.
- Dùng recursive CTE cho truy vấn phân cấp hoặc đệ quy.
- Sử dụng sâu các window function nâng cao.
- Có thể dùng `UNION` hoặc `UNION ALL` để gộp tập kết quả.
- Triển khai logic phức tạp với các hàm phân tích nâng cao.
- Dùng phạm vi rộng các mệnh đề và điều kiện SQL.
- Tận dụng phổ rộng các SQL function và tính năng nâng cao.

**Ví dụ SQL mức Highly Complex:**
```sql
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT employee_id, name, manager_id, department_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.name, e.manager_id, e.department_id, eh.level + 1
    FROM employees e
    JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
),
DepartmentSalaries AS (
    SELECT eh.employee_id, eh.name, eh.level, d.department_name, s.salary, d.department_id
    FROM EmployeeHierarchy eh
    INNER JOIN departments d ON eh.department_id = d.department_id
    INNER JOIN salaries s ON eh.employee_id = s.employee_id
),
DepartmentStats AS (
    SELECT 
        d.department_id,
        COUNT(e.employee_id) AS employee_count,
        AVG(s.salary) AS average_salary
    FROM employees e
    INNER JOIN salaries s ON e.employee_id = s.employee_id
    INNER JOIN departments d ON e.department_id = d.department_id
    GROUP BY d.department_id
)
SELECT ds.name, ds.level, 
    SUM(ds.salary) OVER (PARTITION BY ds.department_id ORDER BY ds.level, ds.name) AS cumulative_salary
FROM DepartmentSalaries ds
INNER JOIN DepartmentStats dstat ON ds.department_id = dstat.department_id
ORDER BY ds.level, ds.name;
```'''

def obtain_db_schema(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_names = []
    create_statements = []
    for table in tables:
        table_name, create_statement = table
        table_names.append(table_name)
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    return table_names, create_statements

def obtain_insert_statements(db_file_dir, table_names):
    table_name2insert_statements = dict()
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    for table_name in table_names:
        try:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 2')
            rows = cursor.fetchall()

            column_names = [description[0] for description in cursor.description]

            insert_statements = []
            for row in rows:
                values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
                insert_statement = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({values});"
                insert_statements.append(insert_statement)

            # for statement in insert_statements:
            #     print(statement)
            table_name2insert_statements[table_name] = insert_statements

        except Exception as e:
            print(e)

    cursor.close()
    conn.close()

    return table_name2insert_statements

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_dbs", type=int, default=0, help="0 means no limit")
    # NOTE: The original implementation sampled a fixed number of SQLs per DB.
    # For better table coverage, we also support sampling per table.
    parser.add_argument(
        "--samples_per_db",
        type=int,
        default=300,
        help="Number of SQL prompts per DB (legacy mode). Ignored when --samples_per_table > 0.",
    )
    parser.add_argument(
        "--samples_per_table",
        type=int,
        default=0,
        help="If >0, generate N SQL prompts for EACH table in a DB. This encourages per-table coverage.",
    )
    opt = parser.parse_args()

    random.seed(42)
    db_path = "../database_synthesis/synthetic_sqlite_databases"
    prompt_template = open("./prompt_templates/sql_synthesis_prompt.txt", "r", encoding = "utf-8").read()
    functions = json.load(open("./prompt_templates/sqlite_funcs.json"))

    complexity2criterion = {
        "Simple": simple_criterion,
        "Moderate": moderate_criterion,
        "Complex": complex_criterion, 
        "Highly Complex": highly_complex_criterion
    }

    db_names = os.listdir(db_path)
    if opt.max_dbs > 0:
        db_names = db_names[:opt.max_dbs]
    prompts = []
    for db_name in tqdm(db_names):
        try:
            db_file_dir = os.path.join(db_path, db_name, db_name + ".sqlite")
            table_names, create_statements = obtain_db_schema(db_file_dir)
            table_name2insert_statements = obtain_insert_statements(db_file_dir, table_names)

            # Decide how many prompts to generate.
            if opt.samples_per_table and opt.samples_per_table > 0:
                table_targets = []
                for tn in table_names:
                    table_targets.extend([tn] * opt.samples_per_table)
            else:
                table_targets = [None] * opt.samples_per_db

            for target_table in table_targets:
                complexity = random.sample(["Simple", "Moderate", "Complex", "Highly Complex"], 1)[0] 

                insert_statements = []
                for table_name in table_names:
                    insert_statements += table_name2insert_statements.get(table_name, [])
                
                if len(insert_statements) == 0:
                    db_value_prompt = ""
                else:
                    if len(insert_statements) > 4:
                        insert_statements = random.sample(insert_statements, 4)
                    db_value_prompt = insert_stmts_template.format(insert_statements = "\n\n".join(insert_statements))

                function_num = random.randint(0, 2)
                if function_num == 0:
                    sql_function_prompt = "### SQL Functions\nBạn có thể dùng bất kỳ function nào được hệ quản trị cơ sở dữ liệu hỗ trợ."
                else:
                    sql_funcs = ""
                    sampled_functions = random.sample(functions, function_num)
                    for idx, func in enumerate(sampled_functions):
                        sql_funcs += f"Function {idx + 1}:\n" + func.strip() + "\n"
                    sql_function_prompt = sql_func_template.format(sql_funcs = sql_funcs)

                column_count = np.random.geometric(0.6, 1)[0]

                # Encourage table coverage: force the SQL to use a specific table.
                if target_table:
                    table_constraint = (
                        "\n\n"
                        "### Ràng buộc bắt buộc về bảng\n"
                        f"- Câu SQL BẮT BUỘC phải sử dụng bảng `{target_table}`.\n"
                        f"- Bảng `{target_table}` phải xuất hiện rõ ràng trong mệnh đề `FROM` hoặc `JOIN`.\n"
                    )
                else:
                    table_constraint = ""

                prompt = (prompt_template + table_constraint).format(
                    schema_str = "\n\n".join(create_statements),
                    sql_function_prompt = sql_function_prompt.strip(),
                    db_value_prompt = db_value_prompt.strip(),
                    complexity = complexity,
                    criterion = complexity2criterion[complexity].strip(),
                    db_engine = "SQLite",
                    column_count = column_count
                )

                prompts.append(
                    {
                        "prompt": prompt,
                        "db_id": db_name,
                        "target_table": target_table,
                        "complexity": complexity,
                        "column_count": int(column_count),
                    }
                )
        except Exception as e:
            print(e)

    with open("./prompts/sql_synthesis_prompts.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(prompts, indent=2, ensure_ascii=False))