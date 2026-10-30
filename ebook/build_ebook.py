"""
Build a single-file HTML research ebook from the project scripts.

Each script in scripts/ becomes a section (in folder and file order). The
module docstring is rendered as the explanation, the source is shown with
syntax highlighting, and the outputs of the matching executed notebook cells
(notebooks/0X_*.ipynb) are shown underneath. Styling, theme and the light/dark
toggle come from ebook_theme.html (the NyayaRAG ebook theme).

Usage:
    python build_ebook.py
"""

import ast
import html
import json
import math
import re
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
NOTEBOOKS_DIR = ROOT / "notebooks"
TEMPLATE = ROOT / "ebook_theme.html"
OUTPUT = ROOT / "Enterprise_Presales_ebook.html"

TITLE = "Enterprise Presales Knowledge Orchestration"
BRAND = "DataMindsLab"

CHAPTERS = [
    ("01_data_loading_cleaning", "01_Data_Loading_Quality_Cleaning.ipynb",
     "Data Loading, Quality Assessment and Cleaning"),
    ("02_exploratory_data_analysis", "02_Exploratory_Data_Analysis.ipynb",
     "Exploratory Data Analysis"),
    ("03_rq1_enterprise_knowledge_requirement", "03_RQ1_Enterprise_Knowledge_Requirement.ipynb",
     "RQ1 – Enterprise Knowledge Requirement"),
    ("04_rq2_knowledge_routing_agent", "04_RQ2_Knowledge_Routing_Agent.ipynb",
     "RQ2 – Knowledge Routing Agent"),
    ("05_rq3_response_generation_agent", "05_RQ3_Response_Generation_Agent.ipynb",
     "RQ3 – Response Generation Agent"),
]

# (notebook file, cell index) -> script file name, for cells the automatic
# alignment places in the wrong section.
MANUAL_OVERRIDES = {
    ("01_Data_Loading_Quality_Cleaning.ipynb", 7): "03_dataset_initial_inspection.py",
    ("03_RQ1_Enterprise_Knowledge_Requirement.ipynb", 35): "10_model_explainability_shap.py",
}

# (notebook file, cell index, output index) -> script file name, for a single
# output of a split cell that no printed banner identifies.
OUTPUT_OVERRIDES = {
    ("02_Exploratory_Data_Analysis.ipynb", 11, 2): "07_predictor_association_heatmap.py",
}

ACRONYMS = {"rq1": "RQ1", "rq2": "RQ2", "rq3": "RQ3", "vif": "VIF", "shap": "SHAP",
            "llm": "LLM", "eda": "EDA", "anova": "ANOVA", "cramers": "Cramér's"}
SMALL_WORDS = {"and", "of", "the", "a", "as", "vs", "to", "in", "for"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COLAB_BADGE_RE = re.compile(r'<a href="https://colab\.research\.google\.com.*?</a>', re.S)
REPO_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/Xquisite-design/[^\s\"'<>]*")
BANNER_RE = re.compile(r"^#{6,}[ \t]*\n((?:#[^\n]*\n)+?)#{6,}[ \t]*$", re.M)
TOKEN_RE =re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOP_TOKENS = {"import", "from", "print", "def", "return", "for", "and", "not", "None",
               "True", "False", "self", "the", "with", "pandas", "numpy", "len"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pretty_title(stem: str) -> str:
    words = stem.split("_")[1:]
    out = []
    for i, w in enumerate(words):
        if w in ACRONYMS:
            out.append(ACRONYMS[w])
        elif i and w in SMALL_WORDS:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_markdown(text: str) -> str:
    text = COLAB_BADGE_RE.sub("", text).strip()
    if not text:
        return ""
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # Keep notebook headings in proportion inside a section.
    body = re.sub(r"<(/?)h[12]>", r"<\1h3>", body)
    body = re.sub(r"<(/?)h[3-6]>", r"<\1h4>", body)
    return body


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", COLAB_BADGE_RE.sub("", text).lower()).strip()


def tokens(text: str) -> set:
    return {t for t in TOKEN_RE.findall(text) if t not in STOP_TOKENS}


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def joined(value) -> str:
    return "".join(value) if isinstance(value, list) else value


# ---------------------------------------------------------------------------
# Notebook-to-script alignment
# ---------------------------------------------------------------------------
def align(code_cells: list, scripts: list) -> list:
    """Assign each code cell a script index, non-decreasing, maximising overlap."""
    script_tokens = [tokens(s["source"]) for s in scripts]
    df = {}
    for toks in script_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n_scripts = len(scripts)
    idf = {t: math.log(1 + n_scripts / c) for t, c in df.items()}

    def score(cell_toks: set, j: int) -> float:
        weights = [idf.get(t, 0.0) for t in cell_toks]
        total = sum(weights)
        if not total:
            return 0.0
        return sum(idf.get(t, 0.0) for t in cell_toks & script_tokens[j]) / total

    scores = [[score(tokens(c["source"]), j) for j in range(n_scripts)] for c in code_cells]
    n = len(code_cells)
    if not n:
        return []
    best = [[-1.0] * n_scripts for _ in range(n)]
    back = [[0] * n_scripts for _ in range(n)]
    for j in range(n_scripts):
        best[0][j] = scores[0][j]
    for i in range(1, n):
        run_max, run_arg = -1.0, 0
        for j in range(n_scripts):
            if best[i - 1][j] > run_max:
                run_max, run_arg = best[i - 1][j], j
            best[i][j] = run_max + scores[i][j]
            back[i][j] = run_arg
    j = max(range(n_scripts), key=lambda k: best[n - 1][k])
    path = [0] * n
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i][j]
    return path


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------
class Counter:
    def __init__(self, chapter_no: int):
        self.chapter_no = chapter_no
        self.figures = 0
        self.tables = 0

    def figure(self) -> str:
        self.figures += 1
        return f"Figure {self.chapter_no}.{self.figures}"

    def table(self) -> str:
        self.tables += 1
        return f"Table {self.chapter_no}.{self.tables}"


class Router:
    """Sends a cell's outputs to the right script when one notebook cell was
    split into several consecutive scripts. Upper-case banner lines printed by
    a later script in the allowed range (e.g. "MISSING VALUES") move output on."""

    def __init__(self, scripts: list, first: int, last: int):
        self.scripts = scripts
        self.current = first
        self.headers = {}
        for k in range(first + 1, last + 1):
            for h in scripts[k]["headers"]:
                self.headers.setdefault(h, k)

    def jump(self, name: str) -> None:
        for k, s in enumerate(self.scripts):
            if s["name"] == name and k > self.current:
                self.current = k

    def emit(self, piece: str) -> None:
        self.scripts[self.current]["outputs"].append(piece)

    def stream(self, text: str) -> None:
        chunk = []
        for line in text.splitlines():
            target = self.headers.get(line.strip())
            if target is not None and target > self.current:
                carried = []
                while chunk and (not chunk[-1].strip() or set(chunk[-1].strip()) <= set("=-")):
                    carried.insert(0, chunk.pop())
                self._flush(chunk)
                chunk = carried
                self.current = target
            chunk.append(line)
        self._flush(chunk)

    def _flush(self, lines: list) -> None:
        text = "\n".join(lines).strip("\n").rstrip()
        if text.strip():
            self.emit(f'<pre class="output">{html.escape(text)}</pre>')


def script_headers(source: str) -> set:
    headers = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return headers
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if 6 <= len(text) <= 80 and "\n" not in text and text.upper() == text and re.search("[A-Z]{3}", text):
                headers.add(text)
    return headers


def render_outputs(nb_name: str, cell: dict, counter: Counter, router: Router) -> None:
    for i, out in enumerate(cell["raw"].get("outputs", [])):
        override = OUTPUT_OVERRIDES.get((nb_name, cell["index"], i))
        if override:
            router.jump(override)
        kind = out.get("output_type")
        if kind == "stream":
            router.stream(ANSI_RE.sub("", joined(out.get("text", ""))))
        elif kind in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                label = counter.figure()
                png = joined(data["image/png"]).replace("\n", "")
                router.emit(f'<figure><img alt="{label}" src="data:image/png;base64,{png}">'
                              f"<figcaption>{label}</figcaption></figure>")
            elif "text/html" in data:
                markup = joined(data["text/html"])
                tables = re.findall(r"<table.*?</table>", markup, re.S)
                if tables:
                    label = counter.table()
                    router.emit(f'<div class="table-wrap"><div class="caption">{label}</div>'
                                  f'<div class="html-output">{"".join(tables)}</div></div>')
            elif "text/plain" in data:
                text = joined(data["text/plain"]).rstrip()
                if text and not re.fullmatch(r"<Figure size .*>", text):
                    router.emit(f'<pre class="output">{html.escape(text)}</pre>')


# ---------------------------------------------------------------------------
# Chapter assembly
# ---------------------------------------------------------------------------
def load_scripts(folder: str) -> list:
    scripts = []
    for path in sorted((SCRIPTS_DIR / folder).glob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            doc = ast.get_docstring(ast.parse(source)) or ""
        except SyntaxError:
            doc = ""
        scripts.append({"path": path, "name": path.name, "source": source, "doc": doc,
                        "title": pretty_title(path.stem), "headers": script_headers(source),
                        "doc_words": set(normalise(doc).split()), "outputs": [], "cells": []})
    return scripts


def is_boilerplate_markdown(text: str) -> bool:
    stripped = COLAB_BADGE_RE.sub("", text).strip()
    return not stripped or bool(re.match(r"#\s*Part \d of 5", stripped))


def is_paraphrased(text: str, scripts: list, current: int) -> bool:
    words = {w for w in normalise(text).split() if len(w) > 3}
    if not words:
        return True
    nearby = scripts[current:current + 2]
    return any(len(words & s["doc_words"]) / len(words) >= 0.6 for s in nearby)


def build_chapter(chapter_no: int, folder: str, nb_name: str, title: str) -> tuple:
    scripts = load_scripts(folder)
    nb = json.loads((NOTEBOOKS_DIR / nb_name).read_text(encoding="utf-8"))
    cells = [{"index": i, "type": c["cell_type"], "source": cell_source(c), "raw": c}
             for i, c in enumerate(nb["cells"])]

    code_cells = [c for c in cells if c["type"] == "code" and c["source"].strip()]
    path = align(code_cells, scripts)
    by_name = {s["name"]: k for k, s in enumerate(scripts)}
    owner = {}
    for cell, j in zip(code_cells, path):
        override = MANUAL_OVERRIDES.get((nb_name, cell["index"]))
        owner[cell["index"]] = by_name[override] if override else j

    all_docs = normalise(" ".join(s["doc"] for s in scripts))
    counter = Counter(chapter_no)
    current = 0
    owned = [c["index"] for c in code_cells]
    for cell in cells:
        if cell["index"] in owner:
            current = owner[cell["index"]]
            # Later scripts with no cell of their own may hold part of this cell's output.
            later = [owner[i] for i in owned if i > cell["index"]]
            last = min(later) - 1 if later else len(scripts) - 1
            router = Router(scripts, current, max(current, last))
            scripts[current]["cells"].append(cell["index"])
            render_outputs(nb_name, cell, counter, router)
            if router.current != current:
                scripts[router.current]["cells"].append(f"{cell['index']}*")
                current = router.current
        elif cell["type"] == "markdown" and not is_boilerplate_markdown(cell["source"]):
            # Skip notes already reproduced (or paraphrased) in a nearby script docstring.
            if normalise(cell["source"]) in all_docs or is_paraphrased(cell["source"], scripts, current):
                continue
            notes = render_markdown(cell["source"])
            if notes:
                scripts[current]["outputs"].append(f'<div class="nb-notes">{notes}</div>')

    print(f"Chapter {chapter_no}: {nb_name}")
    for k, s in enumerate(scripts, 1):
        print(f"  {chapter_no}.{k:<3} {s['name']:<58} cells {s['cells']}")

    formatter = HtmlFormatter(linenos="table")
    sections, toc_items = [], []
    for k, s in enumerate(scripts, 1):
        anchor = f"ch{chapter_no}-{slugify(s['path'].stem)}"
        toc_items.append(f'<li><a href="#{anchor}">{chapter_no}.{k} {html.escape(s["title"])}</a></li>')
        # Code-style "####" banners in docstrings become a bold line, not headings.
        doc = BANNER_RE.sub(lambda m: "**" + " ".join(
            ln.lstrip("#").strip() for ln in m.group(1).splitlines() if ln.lstrip("#").strip()) + "**", s["doc"])
        explanation = render_markdown(doc)
        # Keep project links (Colab badge, GitHub repository) out of the ebook.
        shown = COLAB_BADGE_RE.sub("", s["source"]).replace('"""\n\n', '"""\n', 1)
        shown = REPO_URL_RE.sub("<repository-url>", shown)
        code_html = highlight(shown, PythonLexer(), formatter)
        n_lines = len(shown.splitlines())
        rel = s["path"].relative_to(ROOT).as_posix()
        block = [f'<section class="script" id="{anchor}">',
                 f'<h2><span class="secno">{chapter_no}.{k}</span> {html.escape(s["title"])}</h2>',
                 f'<div class="script-file"><code>{rel}</code></div>']
        if explanation:
            block.append(f'<div class="explanation">{explanation}</div>')
        block.append(f'<details class="code"><summary>Code · {n_lines} lines</summary>{code_html}</details>')
        if s["outputs"]:
            block.append('<div class="results"><h3 class="results-title">Results and outputs</h3>')
            block.extend(s["outputs"])
            block.append("</div>")
        block.append("</section>")
        sections.append("\n".join(block))

    chapter_html = (
        f'<section class="chapter" id="ch{chapter_no}">\n'
        f'<div class="chapter-head"><div class="chapter-label">Chapter {chapter_no}</div>'
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="chapter-source">Source: <code>scripts/{folder}/</code> · executed notebook '
        f"<code>notebooks/{nb_name}</code></p></div>\n" + "\n".join(sections) + "\n</section>"
    )
    toc_html = (f'<li><a href="#ch{chapter_no}">{chapter_no}. {html.escape(title)}</a>'
                f'<ol>{"".join(toc_items)}</ol></li>')
    return chapter_html, toc_html


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    head = template[:template.index("</head>")]
    head = head.replace("<title>NyayaRAG Research Ebook</title>", f"<title>{TITLE} Research Ebook</title>")
    script_start = template.rindex("<script>")
    page_script = template[script_start:template.rindex("</script>") + len("</script>")]
    page_script = page_script.replace("nyayarag-theme", "presales-theme")

    chapters, toc = [], []
    for no, (folder, nb_name, title) in enumerate(CHAPTERS, 1):
        chapter_html, toc_html = build_chapter(no, folder, nb_name, title)
        chapters.append(chapter_html)
        toc.append(toc_html)

    cover_list = "".join(f"<li>{html.escape(t)}</li>" for _, _, t in CHAPTERS)
    page = f"""{head}</head>
<body>
<div class="layout">
<nav class="toc">
  <div class="brand">Presales Agentic RAG</div>
  <div class="muted">{BRAND} / Research</div>
  <ol>{"".join(toc)}</ol>
  <div class="theme"><button id="theme-toggle" type="button">Toggle light / dark</button></div>
</nav>
<main>
<header class="cover">
  <div class="kicker">{BRAND} · Research ebook</div>
  <h1>{TITLE}</h1>
  <p class="subtitle">An agentic retrieval-augmented framework for enterprise presales:
  data quality and exploratory analysis, predicting enterprise knowledge requirements (RQ1),
  routing questions to knowledge repositories (RQ2), and generating and evaluating grounded
  responses with an LLM-as-a-judge (RQ3).</p>
  <ol>{cover_list}</ol>
  <div class="meta">Compiled from the project scripts in <code>scripts/</code>,
  with outputs and figures from the executed notebooks in <code>notebooks/</code>.</div>
</header>
{chr(10).join(chapters)}
<footer class="site-footer"><span>{BRAND} · Research ebook</span><span>{TITLE}</span></footer>
</main>
</div>
{page_script}
</body>
</html>
"""
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"\nWrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
