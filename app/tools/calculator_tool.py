"""
安全计算器工具：用 AST 白名单解析数学表达式，不允许执行任意代码
"""
from __future__ import annotations

import ast
import operator as op

from langchain_core.tools import tool

_SAFE_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}


def _safe_eval(expr: str) -> float:
    """用 AST 白名单解析，只允许数字和四则运算，防止代码注入"""
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式节点：{type(node).__name__}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _eval(tree)


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加(+)减(-)乘(*)除(/)乘方(**)取模(%)。
    示例：'2500 * 12'、'50000 / 22'、'(18000 + 28000) / 2'
    不支持函数调用（如 sqrt、log），仅支持基本四则运算。
    """
    try:
        result = _safe_eval(expression)
        # 整数结果不显示小数点
        if result == int(result):
            return f"{expression} = {int(result)}"
        return f"{expression} = {round(result, 4)}"
    except ZeroDivisionError:
        return "计算错误：除数不能为零。"
    except Exception as e:
        return f"计算失败：{e}。请输入合法的数学表达式，例如 '2500 * 12'。"
