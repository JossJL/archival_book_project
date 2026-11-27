import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typst
import pymupdf
from pydantic import BaseModel

from transcribe_typewritten_script import load_config, resolve_volumes, PipelineConfig


# ==========================================
# 1. Print Trim Presets & Metadata Settings
# ==========================================

@dataclass
class TrimPreset:
    width: str
    height: str
    font_size: str
    leading: str
    gutter_margin: str      # Inside margin (spine edge)
    outer_margin: str       # Fore-edge margin
    top_margin: str
    bottom_margin: str
    first_line_indent: str


TRIM_PRESETS = {
    # Classic UK hardback memoir size (BookVault / Mixam / Lulu)
    "royal": TrimPreset(
        width="156mm",
        height="234mm",
        font_size="10.5pt",
        leading="0.72em",
        gutter_margin="22mm",
        outer_margin="16mm",
        top_margin="20mm",
        bottom_margin="22mm",
        first_line_indent="1.4em",
    ),
    # Compact UK literary hardback / larger paperback
    "demy": TrimPreset(
        width="138mm",
        height="216mm",
        font_size="9.75pt",
        leading="0.68em",
        gutter_margin="19mm",
        outer_margin="14mm",
        top_margin="18mm",
        bottom_margin="20mm",
        first_line_indent="1.3em",
    ),
    # Standard UK commercial paperback (Penguin / Waterstones fiction size)
    "bformat": TrimPreset(
        width="129mm",
        height="198mm",
        font_size="9.5pt",
        leading="0.66em",
        gutter_margin="18mm",
        outer_margin="13mm",
        top_margin="16mm",
        bottom_margin="18mm",
        first_line_indent="1.2em",
    ),
    # 6" x 9" US Trade standard (Lulu / BookVault)
    "ustrade": TrimPreset(
        width="152.4mm",
        height="228.6mm",
        font_size="10.25pt",
        leading="0.70em",
        gutter_margin="21mm",
        outer_margin="15mm",
        top_margin="19mm",
        bottom_margin="21mm",
        first_line_indent="1.35em",
    ),
    # Standard A5 (Mixam / Doxdirect)
    "a5": TrimPreset(
        width="148mm",
        height="210mm",
        font_size="10pt",
        leading="0.70em",
        gutter_margin="20mm",
        outer_margin="15mm",
        top_margin="18mm",
        bottom_margin="20mm",
        first_line_indent="1.3em",
    ),
}


class BookMetadata(BaseModel):
    title: str = "Collected Journals & Memoirs"
    subtitle: str = ""
    author: str = ""
    edition: str = "First Edition"
    publisher_note: str = "Privately printed family edition."
    transcribed_by: str = ""
    edited_by: str = ""


def load_book_metadata(config_path: Path, cfg: PipelineConfig) -> BookMetadata:
    """
    Loads book_metadata from PipelineConfig if present, or directly from
    config.json so you don't even have to modify transcribe_typewritten_script.py.
    """
    if hasattr(cfg, "book_metadata") and cfg.book_metadata is not None:
        if isinstance(cfg.book_metadata, dict):
            return BookMetadata(**cfg.book_metadata)
        return cfg.book_metadata

    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if "book_metadata" in raw and isinstance(raw["book_metadata"], dict):
            return BookMetadata(**raw["book_metadata"])

    return BookMetadata()


# ==========================================
# 2. Bootstrap Clean Manuscript Files for Editing
# ==========================================

def prepare_editable_manuscript(compiled_md: Path, manuscript_md: Path):
    """
    Creates an editable *_manuscript.md copy from *_compiled.md once.
    Strips out the mechanical '---' and '## Page X (Source: ...)' markers.
    """
    if manuscript_md.exists():
        return

    raw = compiled_md.read_text(encoding="utf-8")

    cleaned = re.sub(
        r"\n---\n##\s+Page\s+\d+\s+\*\(Source:\s+.+?\)\*\s*\n?",
        "\n\n",
        raw
    )
    cleaned = re.sub(r"^\*\*(.+?)\*\*\s*$", r"### \1", cleaned, flags=re.MULTILINE)

    guide_banner = (
        "<!-- \n"
        "  EDITABLE MANUSCRIPT FILE (Safe from OCR overwrites)\n"
        "  Formatting Guide:\n"
        "    ## Chapter Title   -> Starts a new Chapter on a fresh page & adds to Table of Contents\n"
        "    ### Subheading     -> Inline section or date heading (no page break)\n"
        "    ***                -> Adds a spaced literary scene break (* * *)\n"
        "    [pagebreak]        -> Forces a manual page break\n"
        "    [footer: text]     -> Renders an unnumbered italic footer at the bottom of the page\n"
        "    ^[footnote text]   -> Renders an inline numbered footnote\n"
        "    > *Note: text*     -> Renders a paragraph-level numbered footnote\n"
        "-->\n\n"
    )
    manuscript_md.write_text(guide_banner + cleaned.strip() + "\n", encoding="utf-8")
    print(f"  [Created Editable Manuscript] {manuscript_md.name}")


# ==========================================
# 3. Encoding Repair, Footers & Markdown Parser
# ==========================================

def clean_encoding_artifacts(text: str) -> str:
    """Repairs common Windows-1252 / UTF-8 mojibake for pound signs, dashes, and quotes."""
    replacements = {
        "Â£": "£",
        "\ufffd": "£",
        "â€”": "—",
        "â€“": "–",
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€\x9d": "”",
        "â€¦": "…",
        "Â ": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Convert inline double hyphens (surrounded by spaces) into clean em-dashes
    text = re.sub(r"(?<=\S)\s+--\s+(?=\S)", " — ", text)
    return text


def jdump(s: str) -> str:
    """UTF-8 safe json.dumps so £, —, and smart quotes stay native Unicode."""
    return json.dumps(s, ensure_ascii=False)


def typst_str(s: str) -> str:
    """Wraps any string as a safe Typst code-mode string literal: #("...");"""
    return f"#({jdump(s)});"


TOKEN_RE = re.compile(
    r"(\*\*[^\*\n]+?\*\*|\*[^\*\n]+?\*|\^\[[^\[\]\n]+?\]|\[[^\[\]\n]{1,80}\])"
)


def md_line_to_typst(line: str) -> str:
    if not line:
        return ""

    parts = TOKEN_RE.split(line)
    out: List[str] = []

    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            inner = part[2:-2]
            out.append(f"#strong({jdump(inner)});")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            inner = part[1:-1]
            out.append(f"#emph({jdump(inner)});")
        elif part.startswith("^[") and part.endswith("]"):
            inner = part[2:-1]
            out.append(f"#footnote[{typst_str(inner)}]")
        elif part.startswith("[") and part.endswith("]"):
            out.append(f"#text(fill: luma(70), {jdump(part)});")
        else:
            out.append(typst_str(part))

    return "".join(out)


def parse_manuscript_to_typst(md_path: Path) -> str:
    try:
        raw_text = md_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = md_path.read_text(encoding="cp1252")

    raw_text = clean_encoding_artifacts(raw_text)

    # Remove HTML comments and mechanical OCR markers
    raw_text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    raw_text = re.sub(r"^#\s+.+?\n", "", raw_text.strip(), count=1)
    raw_text = re.sub(r"^##\s+Page\s+\d+\s+\*\(Source:\s+.+?\)\*\s*$", "", raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r"^---\s*$", "", raw_text, flags=re.MULTILINE)

    # Convert 3+ consecutive newlines (2+ blank lines) into an explicit vertical space token
    raw_text = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n[vspace]\n\n", raw_text.strip())

    paragraphs = re.split(r"\n\s*\n", raw_text)
    typst_blocks: List[str] = []

    note_re = re.compile(r"^>\s*\*(?:Editorial\s+)?Note:\s*(.+?)\*\s*$", re.IGNORECASE | re.DOTALL)
    footer_re = re.compile(r"^\[footer:\s*(.+?)\]\s*$", re.IGNORECASE | re.DOTALL)

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        # 0. Extra Blank Lines (3+ Enters)
        if stripped == "[vspace]":
            typst_blocks.append("\n#v(1.5em)\n")
            continue

        # 1. Manual Page Break: [pagebreak]
        if stripped.lower() == "[pagebreak]":
            typst_blocks.append("\n#pagebreak(weak: true)\n")
            continue

        # 2. Custom Unnumbered Page Footer: [footer: Your footer text here]
        footer_match = footer_re.match(stripped)
        if footer_match:
            footer_text = " ".join(l.strip() for l in footer_match.group(1).splitlines() if l.strip())
            typst_blocks.append(
                f"\n#place(bottom + left, float: true, clearance: 0.8em)["
                f"#block(width: 100%, stroke: (top: 0.4pt + luma(140)), inset: (top: 0.5em))["
                f"#set par(first-line-indent: 0pt)\n"
                f"#text(size: 8.5pt, style: \"italic\", fill: luma(50))[{md_line_to_typst(footer_text)}]"
                f"]]\n"
            )
            continue

        # 3. Literary Scene Break: ***
        if re.match(r"^\*\s*\*\s*\*+$", stripped):
            typst_blocks.append(
                '\n#v(1.2em)#align(center)[#text(size: 10pt, fill: luma(90), "*   *   *")]#v(1.2em)\n'
            )
            continue

        # 4. Chapter Heading: ## Chapter Title
        if stripped.startswith("## "):
            lines = stripped.splitlines()
            chap_title = lines[0][3:].strip()
            typst_blocks.append(
                f"\n#pagebreak(weak: true)\n#v(12%)\n#heading(level: 2, outlined: true)[{md_line_to_typst(chap_title)}]\n#v(1.5em)\n"
            )
            if len(lines) > 1:
                rest_lines = [md_line_to_typst(l.strip()) for l in lines[1:] if l.strip()]
                if rest_lines:
                    typst_blocks.append(" \\\n".join(rest_lines) + "\n")
            continue

        # 5. Sub-heading / Date Header: ### Date or Section
        if stripped.startswith("### "):
            lines = stripped.splitlines()
            sub_title = lines[0][4:].strip()
            typst_blocks.append(
                f"\n#heading(level: 3, outlined: false)[{md_line_to_typst(sub_title)}]\n"
            )
            if len(lines) > 1:
                rest_lines = [md_line_to_typst(l.strip()) for l in lines[1:] if l.strip()]
                if rest_lines:
                    typst_blocks.append(" \\\n".join(rest_lines) + "\n")
            continue

        # 6. Numbered Footnote Block: > *Note: ...*
        note_match = note_re.match(stripped)
        if note_match:
            note_text = " ".join(l.strip() for l in note_match.group(1).splitlines() if l.strip())
            typst_blocks.append(f"#footnote[{md_line_to_typst(note_text)}]\n")
            continue

        # 7. Standard Paragraph (Preserves single Enter line breaks via Typst ' \')
        lines = [md_line_to_typst(line.strip()) for line in stripped.splitlines() if line.strip()]
        if lines:
            typst_blocks.append(" \\\n".join(lines) + "\n")

    return "\n\n".join(typst_blocks)


# ==========================================
# 4. Typst Document Generator (Dynamic Metadata & Presets)
# ==========================================

def generate_typst_source(
    book_title: str,
    subtitle: str,
    author: str,
    edition: str,
    publisher_note: str,
    transcribed_by: str,
    edited_by: str,
    sections: List[tuple[str, str]],
    preset: TrimPreset,
    is_omnibus: bool = False,
) -> str:
    body_parts = []

    for vol_title, rendered_typst_body in sections:
        if is_omnibus:
            body_parts.append(f"""
#pagebreak(to: "odd", weak: true)
#v(28%)
#align(center)[
  #heading(level: 1, outlined: true)[{typst_str(vol_title)}]
  #v(1.2em)
  #line(length: 20%, stroke: 0.5pt + luma(120))
]
#pagebreak(to: "odd", weak: true)
""")
        else:
            body_parts.append(f"""
#v(10%)
#align(center)[
  #heading(level: 1, outlined: false)[{typst_str(vol_title)}]
  #v(1.5em)
]
""")
        body_parts.append(rendered_typst_body)

    joined_body = "\n".join(body_parts)

    # Dynamically build the Copyright / Colophon lines from config.json
    colophon_lines = []
    if edition:
        colophon_lines.append(f"#strong({jdump(edition)});")
    if publisher_note:
        colophon_lines.append(typst_str(publisher_note))

    credit_lines = [
        typst_str("Transcribed and typeset from the original typewritten manuscripts.")
    ]
    if transcribed_by:
        credit_lines.append(typst_str(f"Transcribed by {transcribed_by}"))
    if edited_by:
        credit_lines.append(typst_str(f"Edited by {edited_by}"))

    colophon_top = " \\\n    ".join(colophon_lines)
    colophon_credits = " \\\n    ".join(credit_lines)

    return f"""
// ==========================================
// Auto-Generated Print-Ready Book Layout
// ==========================================

#set document(title: {jdump(f"{book_title} - {subtitle}")}, author: {jdump(author)})

#set text(
  font: ("Georgia", "Liberation Serif", "Times New Roman"),
  size: {preset.font_size},
  fill: luma(0),
  lang: "en",
  region: "gb",
  hyphenate: true,
)

#set page(
  width: {preset.width},
  height: {preset.height},
  margin: (
    inside: {preset.gutter_margin},
    outside: {preset.outer_margin},
    top: {preset.top_margin},
    bottom: {preset.bottom_margin},
  ),
)

#set par(
  justify: true,
  leading: {preset.leading},
  first-line-indent: {preset.first_line_indent},
  spacing: {preset.leading},
)

// Level 1: Volume / Part Divider Title
#show heading.where(level: 1): it => [
  #set text(size: 18pt, weight: "bold", tracking: 1.2pt, fill: luma(0))
  #upper(it.body)
]

// Level 2: Chapter Title (## Chapter ...)
#show heading.where(level: 2): it => block(
  above: 1em,
  below: 1.5em,
  breakable: false,
)[
  #align(center)[
    #set text(size: 14pt, weight: "bold", tracking: 0.5pt, fill: luma(10))
    #it.body
  ]
]

// Level 3: Inline Date / Section Subheading (### ...)
#show heading.where(level: 3): it => block(
  above: 1.6em,
  below: 0.8em,
  breakable: false,
)[
  #set text(size: 11.5pt, weight: "bold", style: "italic", fill: luma(20))
  #it.body
]

// ==========================================
// Front Matter
// ==========================================

#set page(header: none, footer: none)

// 1. Half-Title Page (Recto)
#v(25%)
#align(center)[
  #text(size: 16pt, weight: "regular", tracking: 1.5pt)[{typst_str(book_title.upper())}]
  #v(0.8em)
  #text(size: 12pt, style: "italic")[{typst_str(subtitle)}]
]
#pagebreak()

// 2. Copyright / Edition Verso (Dynamically populated from config.json)
#v(1fr)
#align(left)[
  #set par(first-line-indent: 0pt, leading: 0.65em)
  #text(size: 8.5pt, fill: luma(70))[
    {colophon_top}
    #v(0.8em)
    {colophon_credits}
    #v(0.8em)
    #emph({jdump(subtitle)});
  ]
]
#pagebreak()

// 3. Full Title Page (Recto)
#v(22%)
#align(center)[
  #text(size: 22pt, weight: "bold", tracking: 1pt)[{typst_str(book_title)}]
  #v(1em)
  #text(size: 14pt, style: "italic")[{typst_str(subtitle)}]
  #v(3em)
  #line(length: 30%, stroke: 0.5pt + luma(100))
  #v(3em)
  #text(size: 13pt)[{typst_str(author)}]
]
#pagebreak(to: "odd")

// 4. Table of Contents
#v(15%)
#align(center)[#text(size: 14pt, weight: "bold", tracking: 1pt)[CONTENTS]]
#v(2em)
#outline(title: none, depth: 2, indent: 1.2em)
#pagebreak(to: "odd")

// ==========================================
// Main Narrative Matter (Starts at Page 1)
// ==========================================

#counter(page).update(1)

#set page(
  header: context {{
    let page_num = counter(page).get().first()
    let has_major_heading = query(selector(heading.where(level: 1)).or(heading.where(level: 2))).any(h => h.location().page() == here().page())
    if page_num > 1 and not has_major_heading {{
      set text(size: 8.5pt, style: "italic", fill: luma(70))
      if calc.even(page_num) {{
        align(left)[{typst_str(book_title)}]
      }} else {{
        let past_chaps = query(selector(heading.where(level: 2)).before(here()))
        let past_vols = query(selector(heading.where(level: 1)).before(here()))
        let right_header = if past_chaps.len() > 0 {{
          past_chaps.last().body
        }} else if past_vols.len() > 0 {{
          past_vols.last().body
        }} else {{
          {jdump(subtitle)}
        }}
        align(right)[#right_header]
      }}
    }}
  }},
  footer: context {{
    let page_num = counter(page).get().first()
    let has_part_heading = query(heading.where(level: 1)).any(h => h.location().page() == here().page())
    if not has_part_heading {{
      set text(size: 9.5pt, fill: luma(30))
      if calc.even(page_num) {{
        align(left)[#page_num]
      }} else {{
        align(right)[#page_num]
      }}
    }}
  }}
)

{joined_body}
"""


# ==========================================
# 5. Pre-Press Verification & Page Padding
# ==========================================

def prepress_finalize_pdf(pdf_path: Path):
    doc = pymupdf.open(pdf_path)
    page_count = len(doc)

    remainder = page_count % 4
    added = 0
    if remainder != 0:
        added = 4 - remainder
        width = doc[0].rect.width
        height = doc[0].rect.height
        for _ in range(added):
            doc.new_page(width=width, height=height)

        tmp_pdf = pdf_path.with_suffix(".prepress.pdf")
        doc.save(tmp_pdf, deflate=True, garbage=4)
        doc.close()
        tmp_pdf.replace(pdf_path)
    else:
        doc.close()

    final_total = page_count + added
    spine_80gsm = round(final_total * 0.055, 2)
    spine_100gsm = round(final_total * 0.065, 2)
    print(f"  [Pre-Press Complete] {pdf_path.name}")
    print(f"    -> Final Page Count : {final_total} pages ({added} blank end-pages added for signature alignment)")
    print(f"    -> Est. Spine Width : {spine_80gsm} mm (80gsm paper) | {spine_100gsm} mm (90-100gsm cream bookwove)")
    print(f"    -> Colour Space     : Monochrome Luma / Grayscale")


# ==========================================
# 6. CLI Entry Point
# ==========================================

def build_books(
    cfg: PipelineConfig,
    meta: BookMetadata,
    book_title: Optional[str],
    author: Optional[str],
    trim: str,
    omnibus: bool,
):
    print_dir = cfg.output_dir / "print_ready_pdfs"
    print_dir.mkdir(parents=True, exist_ok=True)

    final_title = book_title or meta.title
    final_author = author if author is not None else meta.author

    preset = TRIM_PRESETS[trim]

    # Resolve volumes directly from config.volumes without requiring input_pdfs/ to exist on disk.
    # If config.volumes is empty, auto-discover any *_manuscript.md or *_compiled.md in output_dir!
    if cfg.volumes:
        volume_specs = [(v.title, v.output_prefix) for v in cfg.volumes]
    else:
        discovered = sorted(cfg.output_dir.glob("*_manuscript.md")) or sorted(cfg.output_dir.glob("*_compiled.md"))
        volume_specs = [
            (p.stem.replace("_manuscript", "").replace("_compiled", "").replace("_", " "),
             p.stem.replace("_manuscript", "").replace("_compiled", ""))
            for p in discovered
        ]

    loaded_sections: List[tuple[str, str, str]] = []
    for vol_title, file_prefix in volume_specs:
        compiled_md = cfg.output_dir / f"{file_prefix}_compiled.md"
        manuscript_md = cfg.output_dir / f"{file_prefix}_manuscript.md"

        if not compiled_md.exists() and not manuscript_md.exists():
            print(f"[Skipping] Neither {manuscript_md.name} nor {compiled_md.name} found in {cfg.output_dir}/")
            continue

        if compiled_md.exists() and not manuscript_md.exists():
            prepare_editable_manuscript(compiled_md, manuscript_md)

        source_md = manuscript_md if manuscript_md.exists() else compiled_md
        print(f"  Reading manuscript from: {source_md.name}")
        rendered_typst = parse_manuscript_to_typst(source_md)
        loaded_sections.append((vol_title, file_prefix, rendered_typst))

    if not loaded_sections:
        print("No Markdown volumes found to typeset.")
        return

    if omnibus:
        subtitle_str = meta.subtitle if meta.subtitle else " · ".join(sec[0] for sec in loaded_sections)
        print(f"\nTypesetting OMNIBUS EDITION ({subtitle_str}) [{trim.upper()}: {preset.width} x {preset.height}]...")

        sections_data = [(sec[0], sec[2]) for sec in loaded_sections]
        typ_source = generate_typst_source(
            book_title=final_title,
            subtitle=subtitle_str,
            author=final_author,
            edition=meta.edition,
            publisher_note=meta.publisher_note,
            transcribed_by=meta.transcribed_by,
            edited_by=meta.edited_by,
            sections=sections_data,
            preset=preset,
            is_omnibus=True,
        )

        typ_path = print_dir / "Omnibus_Edition_print.typ"
        pdf_path = print_dir / "Omnibus_Edition_PRINT_READY.pdf"
        typ_path.write_text(typ_source, encoding="utf-8")
        try:
            typst.compile(str(typ_path), output=str(pdf_path))
        except Exception as e:
            print(f"\n[!] Typst Compilation Failed on {typ_path.name}")
            print(f"    Inspect the generated source file at: {typ_path.resolve()}")
            print(f"    Error details: {e}")
            raise
        prepress_finalize_pdf(pdf_path)
        return

    for vol_title, file_prefix, rendered_typst in loaded_sections:
        print(f"\nTypesetting {vol_title} ({trim.upper()}: {preset.width} x {preset.height})...")
        typ_source = generate_typst_source(
            book_title=final_title,
            subtitle=vol_title,
            author=final_author,
            edition=meta.edition,
            publisher_note=meta.publisher_note,
            transcribed_by=meta.transcribed_by,
            edited_by=meta.edited_by,
            sections=[(vol_title, rendered_typst)],
            preset=preset,
            is_omnibus=False,
        )

        typ_path = print_dir / f"{file_prefix}_print.typ"
        pdf_path = print_dir / f"{file_prefix}_PRINT_READY.pdf"
        typ_path.write_text(typ_source, encoding="utf-8")
        try:
            typst.compile(str(typ_path), output=str(pdf_path))
        except Exception as e:
            print(f"\n[!] Typst Compilation Failed on {typ_path.name}")
            print(f"    Inspect the generated source file at: {typ_path.resolve()}")
            print(f"    Error details: {e}")
            raise
        prepress_finalize_pdf(pdf_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Volume Manuscript files into print-ready book PDFs.")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.json"), help="Path to config.json")
    parser.add_argument("--title", type=str, default=None, help="Override book_metadata.title from config.json")
    parser.add_argument("--author", type=str, default=None, help="Override book_metadata.author from config.json")
    parser.add_argument(
        "--trim",
        choices=list(TRIM_PRESETS.keys()),
        default="royal",
        help="Book trim size & typography preset (royal, demy, bformat, ustrade, a5)",
    )
    parser.add_argument("--omnibus", action="store_true", help="Combine volumes into a single Omnibus Edition PDF")
    args = parser.parse_args()

    config = load_config(args.config)
    book_meta = load_book_metadata(args.config, config)

    build_books(
        cfg=config,
        meta=book_meta,
        book_title=args.title,
        author=args.author,
        trim=args.trim,
        omnibus=args.omnibus,
    )