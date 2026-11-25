"""Turn pandoc's longtables into non-breaking floats held in place ([H]).

The report's tables are short, so a page break inside one only separates rows
from their header. Usage: python fix_tables.py REPORT.tex (edits in place).
"""
import re
import sys

path = sys.argv[1]
tex = open(path, encoding="utf-8").read()
pat = re.compile(r"\\begin\{longtable\}\[\]\{(@\{\}.*?@\{\})\}\n(.*?)\\end\{longtable\}", re.S)


def convert(m):
    spec, body = m.group(1), m.group(2)
    caption = ""
    cm = re.match(r"\\caption\{(.*?)\}\\tabularnewline\n", body, re.S)
    if cm:
        caption, body = cm.group(1), body[cm.end():]
    if "\\endfirsthead" in body:
        head, rest = body.split("\\endfirsthead", 1)
        rest = rest.split("\\endhead", 1)[1]
    else:
        head, rest = body.split("\\endhead", 1)
    foot, rows = rest.split("\\endlastfoot", 1)
    out = "\\begin{table}[htbp]\n\\centering\\small\n"
    if caption:
        out += "\\caption{" + caption + "}\n"
    out += "\\begin{tabular}{" + spec + "}\n" + head.strip() + "\n" + rows.strip() + "\n" + foot.strip() + "\n\\end{tabular}\n\\end{table}"
    return out


new, n = pat.subn(convert, tex)
open(path, "w", encoding="utf-8").write(new)
print(f"converted {n} tables")
