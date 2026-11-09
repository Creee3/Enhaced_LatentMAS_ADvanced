"""Build the self-contained offline simulator from its editable source files."""
from pathlib import Path

root = Path(__file__).resolve().parent
page = (root / "index.template.html").read_text(encoding="utf-8")
for filename, placeholder in [
    ("style.css", '<link rel="stylesheet" href="style.css">'),
    ("engine.js", '<script src="engine.js"></script>'),
    ("app.js", '<script src="app.js"></script>'),
]:
    tag = "style" if filename.endswith(".css") else "script"
    content = (root / filename).read_text(encoding="utf-8")
    assert page.count(placeholder) == 1, placeholder
    page = page.replace(placeholder, f"<{tag}>\n{content}\n</{tag}>")
(root / "ggg-rrr-simulator.html").write_text(page, encoding="utf-8")
