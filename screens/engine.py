import pandas as pd
from typing import Dict

def eval_rule(df: pd.DataFrame, expr: str) -> pd.Series:
    expr = expr.replace(' between ', ' _between_ ')
    if '_between_' in expr:
        left, rest = expr.split('_between_')
        lo_str, hi_str = rest.split('and')
        lo = float(lo_str.strip())
        hi = float(hi_str.strip())
        return (df[left.strip()] >= lo) & (df[left.strip()] <= hi)
    return pd.eval(expr, engine='python', parser='pandas', local_dict={c: df[c] for c in df.columns})

def run_screen(df: pd.DataFrame, yaml_cfg: Dict) -> pd.DataFrame:
    rules = yaml_cfg.get('rules', [])
    out = df.copy()
    for r in rules:
        m = eval_rule(out, r['expr'])
        out = out[m]
    for key in yaml_cfg.get('sort', []):
        ascending = not key.startswith('-')
        col = key.lstrip('+-')
        if col in out.columns:
            out = out.sort_values(col, ascending=ascending)
    lim = yaml_cfg.get('limit')
    if lim:
        out = out.head(int(lim))
    return out
