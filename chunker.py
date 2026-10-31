import json, re, pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ---- change these two for each PDF ----
pdf = "acme.pdf"
experiment = "ACME"
# ---------------------------------------

md = pymupdf4llm.to_markdown(pdf)

# 1. remove junk
md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
md = re.sub(r"</?u>|<br>", "", md)
# 2. bold-only lines become headings
md = re.sub(r"^\*\*(.+?)\*\*\s*$", r"## \1", md, flags=re.M)
# 3. rejoin sentences broken by page breaks
md = re.sub(r"(\w)-\s*\n{1,2}(?=[a-z])", r"\1-", md)
md = re.sub(r"(?<![.!?:\n#*])\s*\n{1,2}(?=[a-z])", " ", md)

open("debug.md", "w", encoding="utf-8").write(md)   # to inspect the cleaned text

by_header = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
).split_text(md)

docs = RecursiveCharacterTextSplitter(
    chunk_size=1800, chunk_overlap=150
).split_documents(by_header)

clean = lambda s: s.replace("**", "").strip()
SKIP = ("links", "status", "contact")
chunks, parent, n = [], "", 0
for d in docs:
    body = re.sub(r"https?://\S+", "", d.page_content.replace("**", "")).strip()
    sec = clean(d.metadata.get("h3") or d.metadata.get("h2") or "overview")
    if len(body) < 120 or any(w in sec.lower() for w in SKIP):
        continue
    if sec.isupper():
        sub = f"{parent} > {sec}" if parent else sec
    else:
        parent, sub = sec, sec
    n += 1
    chunks.append({
        "id": f"{experiment.lower()}_{n:03d}",
        "experiment": experiment,
        "subtopic": sub,
        "text": f"[{experiment} | {sub}]\n{body}",
    })

json.dump(chunks, open(f"chunks_{experiment.lower()}.json", "w"), indent=2)
print(len(chunks), "chunks written to", f"chunks_{experiment.lower()}.json")