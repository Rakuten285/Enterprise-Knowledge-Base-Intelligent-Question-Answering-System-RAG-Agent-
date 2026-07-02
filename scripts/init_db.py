"""
初始化员工数据库（SQLite）

用法：
    python scripts/init_db.py

会在 data/employees.db 创建 employees 表并插入 10 条示例数据。
字段：id, name, department, position, salary（月薪，元）, hire_date
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.db"

EMPLOYEES = [
    (1, "张伟",   "技术部",   "高级工程师",   28000, "2020-03-15"),
    (2, "李娜",   "技术部",   "工程师",       18000, "2022-07-01"),
    (3, "王芳",   "人力资源部", "HR经理",      15000, "2019-11-20"),
    (4, "刘洋",   "财务部",   "财务专员",     12000, "2021-05-10"),
    (5, "陈静",   "市场部",   "市场总监",     35000, "2018-08-08"),
    (6, "赵磊",   "技术部",   "架构师",       45000, "2017-02-14"),
    (7, "周婷",   "运营部",   "运营专员",     10000, "2023-01-18"),
    (8, "吴强",   "采购部",   "采购经理",     20000, "2020-09-30"),
    (9, "徐敏",   "市场部",   "市场专员",     11000, "2022-03-22"),
    (10,"孙浩",   "技术部",   "初级工程师",    12000, "2023-06-05"),
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS employees (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    department  TEXT    NOT NULL,
    position    TEXT    NOT NULL,
    salary      INTEGER NOT NULL,
    hire_date   TEXT    NOT NULL
);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO employees (id, name, department, position, salary, hire_date)
VALUES (?, ?, ?, ?, ?, ?);
"""


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(CREATE_SQL)
    cursor.executemany(INSERT_SQL, EMPLOYEES)
    conn.commit()
    conn.close()
    print(f"[OK] 员工数据库初始化完成：{DB_PATH}")
    print(f"   共 {len(EMPLOYEES)} 条员工记录")


if __name__ == "__main__":
    main()
