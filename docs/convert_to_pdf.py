"""Convert PAPER_DRAFT.md to a styled PDF using weasyprint."""
import markdown
from pathlib import Path
from weasyprint import HTML

src = Path(__file__).parent / "PAPER_DRAFT.md"
out = Path(__file__).parent / "PAPER_DRAFT.pdf"

md_text = src.read_text(encoding="utf-8")
body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 2.5cm; }}
  body {{ font-family: 'Georgia', 'Times New Roman', serif; font-size: 11pt; line-height: 1.6; color: #222; }}
  h1 {{ font-size: 16pt; margin-top: 0; color: #1a1a1a; text-align: center; }}
  h2 {{ font-size: 13pt; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 3pt; margin-top: 20pt; }}
  h3 {{ font-size: 11.5pt; color: #444; margin-top: 14pt; }}
  p {{ text-align: justify; margin: 6pt 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 10pt; }}
  th, td {{ border: 1px solid #ccc; padding: 4pt 8pt; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  code {{ font-family: 'Consolas', monospace; font-size: 9.5pt; background: #f7f7f7; padding: 1pt 3pt; }}
  pre {{ background: #f7f7f7; padding: 8pt; font-size: 9pt; overflow-x: auto; }}
  strong {{ color: #1a1a1a; }}
  ol, ul {{ margin: 6pt 0; padding-left: 24pt; }}
  li {{ margin: 3pt 0; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 16pt 0; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

HTML(string=html).write_pdf(str(out))
print(f"PDF written to {out}")
print(f"Size: {out.stat().st_size / 1024:.0f} KB")
