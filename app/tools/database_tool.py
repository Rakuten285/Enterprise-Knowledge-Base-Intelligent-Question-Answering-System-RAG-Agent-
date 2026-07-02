"""
员工数据库查询工具：只读 SQLite，防止任意写操作
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.tools import tool

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "employees.db"


@tool
def query_employee_database(sql: str) -> str:
    """查询员工数据库，获取员工的姓名、部门、职位、月薪、入职日期等信息。
    数据库表名：employees
    字段：id(编号), name(姓名), department(部门), position(职位), salary(月薪元), hire_date(入职日期)
    只允许 SELECT 查询。示例：SELECT name, salary FROM employees WHERE department='技术部'
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "安全限制：只允许 SELECT 查询，不允许修改数据。"

    if not _DB_PATH.exists():
        return "员工数据库不存在，请先运行：python scripts/init_db.py"

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        return f"SQL 执行错误：{e}"

    if not rows:
        return "查询结果为空。"

    headers = list(rows[0].keys())
    lines = [" | ".join(headers), "-" * (len(" | ".join(headers)))]
    for row in rows:
        lines.append(" | ".join(str(row[h]) for h in headers))
    return "\n".join(lines)
