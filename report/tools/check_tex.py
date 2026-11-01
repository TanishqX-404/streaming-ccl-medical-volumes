"""Static sanity checks for main.tex (no LaTeX installation needed) + rough page estimate."""
import os
import re
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = open(os.path.join(ROOT, "main.tex"), encoding="utf-8").read()
body = re.sub(r"(?m)(?<!\\)%.*$", "", src)          # strip comments
problems = []

# 1. environments balanced
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, name = m.groups()
    if kind == "begin":
        stack.append(name)
    else:
        if not stack or stack[-1] != name:
            problems.append(f"environment mismatch at \\end{{{name}}} (open: {stack[-3:]})")
            break
        stack.pop()
if stack:
    problems.append(f"unclosed environments: {stack}")

# 2. braces balanced (ignoring escaped braces and verbatim-like listings)
b2 = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", body, flags=re.S)
b2 = b2.replace("\\{", "").replace("\\}", "")
depth = 0
for i, ch in enumerate(b2):
    depth += ch == "{"
    depth -= ch == "}"
    if depth < 0:
        problems.append("negative brace depth near: " + b2[max(0, i - 60):i + 20].replace("\n", " "))
        break
if depth != 0:
    problems.append(f"brace depth at end: {depth}")

# 3. labels / refs
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", body)) - {"LastPage"}
for r in sorted(refs - labels):
    problems.append(f"undefined ref: {r}")
for l in sorted(labels - refs):
    problems.append(f"label never referenced (ok but check): {l}")

# 4. citations
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", body):
    cites.update(x.strip() for x in m.group(1).split(","))
for c in sorted(cites - bib):
    problems.append(f"undefined cite: {c}")
for c in sorted(bib - cites):
    problems.append(f"uncited reference: {c}")

# 5. graphics exist
fig_h = {}
for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    name = m.group(1)
    cands = [os.path.join(ROOT, name), os.path.join(ROOT, "figures", name)]
    found = [c for c in cands if os.path.exists(c)]
    if not found:
        problems.append(f"missing graphic: {name}")
        continue
    png = os.path.splitext(found[0])[0] + ".png"
    if os.path.exists(png):
        im = Image.open(png)
        fig_h[name] = im.height / im.width

# 6. tabular column counts
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}(.*?)\\end\{tabular\}", body, flags=re.S):
    spec, rows = m.group(1), m.group(2)
    spec_clean = re.sub(r"@\{[^}]*\}", "", spec)
    spec_clean = re.sub(r"p\{[^}]*\}", "p", spec_clean)
    ncol = len(re.findall(r"[lrcp]", spec_clean))
    for row in re.split(r"\\\\", rows):
        row = re.sub(r"\\(toprule|midrule|bottomrule|hrow|rowcolor\{[^}]*\})", "", row).strip()
        if not row:
            continue
        n = len(re.findall(r"(?<!\\)&", row)) + 1
        if n != ncol:
            problems.append(f"tabular with {ncol} cols has row with {n}: {row[:70]!r}")

# 7. non-ascii characters (inputenc utf8 is on, but keep source portable)
for i, line in enumerate(src.splitlines(), 1):
    if any(ord(c) > 127 for c in line):
        problems.append(f"non-ASCII on line {i}: {line[:60]!r}")

# 8. rough page estimate (A4, 17.5 x 22.7 cm text block)
page_cm = 22.7
text = re.sub(r"\\begin\{(figure|table|lstlisting|thebibliography).*?\\end\{\1\}", "", body, flags=re.S)
main = text.split("\\section{Introduction}")[1].split("\\begin{thebibliography}")[0] if "\\section{Introduction}" in text else text
main = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", main)
words = len(re.findall(r"[A-Za-z0-9]{2,}", main))
text_cm = words / 15.5 * 0.50 * 1.0
n_sec = len(re.findall(r"\\section\{", body)) - 0
n_sub = len(re.findall(r"\\subsection\{", body))
head_cm = n_sec * 1.5 + n_sub * 1.0
fig_cm = 0
for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", body, flags=re.S):
    inc = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", m.group(0)).group(1)
    cap = re.search(r"\\caption\{(.*?)\}\\label|\\caption\{(.*?)\}\s*\\end", m.group(0), flags=re.S)
    caplen = len(cap.group(0)) if cap else 150
    fig_cm += 17.5 * fig_h.get(inc, 0.35) + 0.45 * (1 + caplen // 105) + 0.5
tab_cm = 0
for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", body, flags=re.S):
    rows = m.group(0).count("\\\\")
    cap = re.search(r"\\caption\{(.*?)\}", m.group(0), flags=re.S)
    tab_cm += rows * 0.56 + 0.45 * (1 + (len(cap.group(0)) if cap else 100) // 105) + 0.7
lst = re.search(r"\\begin\{lstlisting\}\[caption.*?\\end\{lstlisting\}", body, flags=re.S)
lst_cm = (lst.group(0).count("\n") * 0.40 + 1.2) if lst else 0
body_cm = text_cm + head_cm + fig_cm + tab_cm + lst_cm
front = 4
refs_cm = len(bib) * 1.0 + 1.5
app_cm = 14
total = front + body_cm / page_cm / 0.93 + (refs_cm + app_cm) / page_cm + 0.4
print(f"words(body)={words}  text={text_cm:.0f}cm  headings={head_cm:.0f}cm  figures={fig_cm:.0f}cm "
      f"tables={tab_cm:.0f}cm  listing={lst_cm:.0f}cm")
print(f"estimated pages: body {body_cm / page_cm / 0.93:.1f} + front 4 + refs/appendix "
      f"{(refs_cm + app_cm) / page_cm:.1f}  => ~{total:.1f}")
print("figures:", sum(1 for _ in re.finditer(r"\\begin\{figure\}", body)),
      " tables:", sum(1 for _ in re.finditer(r"\\begin\{table\}", body)))
print("\nPROBLEMS:" if problems else "\nno structural problems found")
for p in problems:
    print(" -", p)
sys.exit(0)
