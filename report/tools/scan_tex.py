import re
src = open(r"C:\Users\tanis\Desktop\DS_Project\report\main.tex", encoding="utf-8").read().splitlines()
in_lst = False
for i, line in enumerate(src, 1):
    if "\\begin{lstlisting}" in line:
        in_lst = True
    if in_lst:
        if "\\end{lstlisting}" in line:
            in_lst = False
        continue
    if line.lstrip().startswith("%"):
        continue
    # remove math and known safe constructs
    t = re.sub(r"\$[^$]*\$", "", line)
    t = re.sub(r"\\(label|ref|cite|bibitem|includegraphics|url|href)(\[[^\]]*\])?\{[^}]*\}", "", t)
    if re.search(r"(?<!\\)_", t):
        print("underscore", i, line.strip()[:100])
    if re.search(r"(?<!\\)%", t):
        print("percent", i, line.strip()[:100])
    if re.search(r"(?<!\\)#", t) and "newcommand" not in t:
        print("hash", i, line.strip()[:100])
    if "&" in t and "\\\\" not in t and "tabular" not in t and "\\&" not in t and "begin" not in t:
        if re.search(r"(?<!\\)&", t):
            print("ampersand-outside-table?", i, line.strip()[:100])
print("scan done")
