import os
import sys
import json
import time
import random
import sqlite3
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load GEMINI_API_KEY from .env file
load_dotenv()

MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 15


# ==========================================
# 1. Configuration & Data Schemas
# ==========================================

class VolumeSpec(BaseModel):
    title: str
    output_prefix: str
    pdfs: List[str]


class BookMetadata(BaseModel):
    title: str = "Collected Journals & Memoirs"
    subtitle: str = ""
    author: str = ""
    edition: str = "First Edition"
    publisher_note: str = "Privately printed family edition."
    transcribed_by: str = ""
    edited_by: str = ""

class PipelineConfig(BaseModel):
    model: str = "gemini-3.8-flash"
    dpi: int = 300
    max_allowed_tokens: int = 500_000
    input_dir: Path = Path("./input_pdfs")
    clean_img_dir: Path = Path("./cleaned_pages")
    output_dir: Path = Path("./transcriptions")
    author_context: str = "British English spelling and idioms."
    known_entities: List[str] = Field(default_factory=list)
    book_metadata: BookMetadata = Field(default_factory=BookMetadata)
    volumes: List[VolumeSpec] = Field(default_factory=list)

    def build_system_prompt(self) -> str:
        entities_str = ", ".join(self.known_entities) if self.known_entities else "None specified"
        return f"""
You are an expert archivist transcribing 20th-century typewritten pages that may contain handwritten corrections, margin notes, or full handwritten sections.
Author context: {self.author_context}
Known family names / locations to watch out for: {entities_str}.

Rules:
1. Typewriter Quirks: Silently normalize mechanical typewriter substitutions where clearly intended (e.g., lowercase 'l' typed for the numeral '1', or capital 'O' typed for '0'), and rejoin words split across lines by end-of-line hyphens.
2. Strikeouts & Overstrikes: Ignore words that were intentionally X'd out, slashed through (e.g., w/o/r/d), or crossed out by pen, unless historically significant (if so, log in archivist_notes).
3. Handwritten Edits: If a handwritten word is inserted above a line (with a caret ^) or corrects a typed word, incorporate the correction seamlessly. If there is a separate margin note, include it and note it in archivist_notes.
4. Faded Ribbon / Illegible Words: Do NOT silently guess unreadable words. Write your best guess in brackets with a question mark, like [Basingstoke?], and log it in uncertain_words.
# Inside build_system_prompt() in transcribe_journals.py:
5. Page Numbers: Do NOT transcribe standalone page numbers or folio numbers appearing at the top or bottom margins of the original sheets (e.g., "14", "- 14 -", "Page 14").
""".strip()


class UncertainWord(BaseModel):
    guess: str = Field(description="Your best guess at the word or phrase.")
    line_context: str = Field(description="The surrounding sentence or line where it appears.")
    alternatives: List[str] = Field(description="Other plausible readings of the ink strokes.")

class PageTranscription(BaseModel):
    date_or_heading: Optional[str] = Field(description="Any date, location, or chapter title at the top of the page. Do NOT include page numbers here.")
    transcription: str = Field(description="The full Markdown transcription of the page.")
    uncertain_words: List[UncertainWord] = Field(description="List of words that were difficult to decipher.")
    archivist_notes: Optional[str] = Field(description="Notes on sketches, crossed-out sections, or margin notes.")


def load_config(config_path: Path) -> PipelineConfig:
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return PipelineConfig(**raw)
        
    # If config.json doesn't exist yet, auto-create it from config.example.json
    example_path = Path("config.example.json")
    if config_path == Path("config.json") and example_path.exists():
        shutil.copy(example_path, config_path)
        print(f"[Setup] Created a local '{config_path}' from '{example_path}'.")
        print("        Edit 'config.json' to add your custom family names or volume groupings.")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return PipelineConfig(**raw)

    print(f"[Info] No {config_path} found; using built-in default configuration.")
    return PipelineConfig()


def resolve_volumes(cfg: PipelineConfig) -> List[tuple[str, str, List[Path]]]:
    """
    Returns a list of (Volume Title, Output File Prefix, List[Path to PDFs]).
    If no volumes are defined in config.json, defaults to 1 volume per PDF in input_dir.
    """
    if cfg.volumes:
        resolved = []
        for vol in cfg.volumes:
            paths = [cfg.input_dir / pdf_name for pdf_name in vol.pdfs if (cfg.input_dir / pdf_name).exists()]
            if paths:
                resolved.append((vol.title, vol.output_prefix, paths))
            else:
                print(f"[Warning] No matching PDFs found in {cfg.input_dir} for {vol.title}")
        return resolved

    # Fallback: Every PDF in input_dir becomes its own standalone volume
    all_pdfs = sorted(cfg.input_dir.glob("*.pdf"))
    return [(p.stem.replace("_", " ").title(), p.stem, [p]) for p in all_pdfs]


# ==========================================
# 2. Idempotency & State Ledger (SQLite)
# ==========================================

def init_db(output_dir: Path) -> sqlite3.Connection:
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_dir / "pipeline_state.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS page_progress (
            journal_name TEXT,
            page_num INTEGER,
            page_id TEXT,
            image_cleaned INTEGER DEFAULT 0,
            transcribed INTEGER DEFAULT 0,
            prompt_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            tail_context TEXT DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (journal_name, page_num)
        )
    """)
    conn.commit()
    return conn


def atomic_write_text(target_path: Path, content: str):
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, target_path)


def atomic_write_image(target_path: Path, img_array: np.ndarray):
    tmp_path = target_path.with_name(target_path.stem + "_tmp.png")
    if not cv2.imwrite(str(tmp_path), img_array):
        raise IOError(f"Failed to encode image for {target_path}")
    os.replace(tmp_path, target_path)


# ==========================================
# 3. Single-Page Extraction & OpenCV Cleanup
# ==========================================

def render_single_page(doc: fitz.Document, page_index: int, dpi: int = 300) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def deskew_image(gray_img: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(gray_img)
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 500:
        return gray_img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if 0.2 < abs(angle) < 10.0:
        (h, w) = gray_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return gray_img


def clean_photocopy(gray_img: np.ndarray) -> np.ndarray:
    straightened = deskew_image(gray_img)
    kernel_size = max(51, int(min(straightened.shape) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(straightened, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(straightened, background, scale=255)
    denoised = cv2.fastNlMeansDenoising(normalized, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)
    _, cleaned = cv2.threshold(contrast_enhanced, 225, 255, cv2.THRESH_TRUNC)
    return cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)


# ==========================================
# 4. Fault-Tolerant Gemini Transcription
# ==========================================

class QuotaExhaustedException(Exception):
    pass


def transcribe_page_with_retry(
    client: genai.Client,
    cfg: PipelineConfig,
    image_path: Path,
    prev_context: str = ""
) -> tuple[PageTranscription, int, int]:
    pil_img = Image.open(image_path)
    prompt = "Transcribe this page."
    if prev_context:
        prompt += (
            f"\n\nFor continuity, here are the final lines from the previous page "
            f"(in case a sentence carries over):\n\"...{prev_context}\""
        )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=cfg.model,
                contents=[pil_img, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=cfg.build_system_prompt(),
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=PageTranscription,
                ),
            )
            usage = response.usage_metadata
            in_tokens = usage.prompt_token_count if usage else 0
            out_tokens = usage.candidates_token_count if usage else 0
            return PageTranscription.model_validate_json(response.text), in_tokens, out_tokens

        except APIError as e:
            err_msg = str(e).lower()
            status_code = getattr(e, "code", None)
            if status_code == 429 and ("per day" in err_msg or "billing" in err_msg or "credit" in err_msg):
                raise QuotaExhaustedException(f"Hard quota/token limit reached ({e}). Re-run when quota resets.")

            if status_code in (429, 500, 502, 503, 504):
                if attempt == MAX_RETRIES:
                    raise QuotaExhaustedException(f"Max retries ({MAX_RETRIES}) exceeded: {e}")
                wait_time = (BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))) + random.uniform(1, 5)
                print(f"    [Rate Limit {status_code}] Waiting {wait_time:.1f}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait_time)
            else:
                raise


# ==========================================
# 5. Compilation & Interactive Review
# ==========================================

def compile_all_volumes(
    cfg: PipelineConfig,
    volume_groups: List[tuple[str, str, List[Path]]],
    conn: sqlite3.Connection,
    include_notes: bool = False
):
    for vol_title, file_prefix, pdf_list in volume_groups:
        md_lines = [f"# {vol_title}\n"]
        review_lines = [f"# Proofreading Log: {vol_title}\n"]
        vol_page_counter = 1
        unresolved_count = 0

        for pdf_path in pdf_list:
            journal_name = pdf_path.stem
            json_dir = cfg.output_dir / journal_name
            img_dir = cfg.clean_img_dir / journal_name
            if not json_dir.exists():
                continue

            for json_file in sorted(json_dir.glob("page_*.json")):
                page_id = json_file.stem
                data = PageTranscription.model_validate_json(json_file.read_text(encoding="utf-8"))

                md_lines.append(f"\n---\n## Page {vol_page_counter} *(Source: {journal_name} / {page_id})*")
                if data.date_or_heading:
                    md_lines.append(f"**{data.date_or_heading}**\n")
                md_lines.append(data.transcription)

                # Only include archivist_notes in the book if --with-notes was passed
                if include_notes and data.archivist_notes:
                    md_lines.append(f"\n> *Editorial Note: {data.archivist_notes}*")

                if data.uncertain_words:
                    review_lines.append(f"\n### Vol Page {vol_page_counter} — `{img_dir / (page_id + '.png')}`")
                    review_lines.append(f"*JSON file: `{json_file}`*")
                    for item in data.uncertain_words:
                        unresolved_count += 1
                        alts = ", ".join(item.alternatives) if item.alternatives else "None"
                        review_lines.append(
                            f"- **`{item.guess}`** (Alternatives: *{alts}*)\n"
                            f"  - Context: \"{item.line_context}\""
                        )
                vol_page_counter += 1

        if unresolved_count == 0:
            review_lines.append("\n*All uncertain words in this volume have been resolved!*")

        atomic_write_text(cfg.output_dir / f"{file_prefix}_compiled.md", "\n".join(md_lines))
        atomic_write_text(cfg.output_dir / f"{file_prefix}_needs_review.md", "\n".join(review_lines))
        print(f"  -> Compiled {file_prefix}_compiled.md ({unresolved_count} uncertain words remaining)")

    stats = conn.execute(
        "SELECT COUNT(*), SUM(transcribed), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(output_tokens),0) FROM page_progress"
    ).fetchone()
    print(f"\n[Summary] {stats[1] or 0}/{stats[0] or 0} total pages transcribed | "
          f"Tokens: {stats[2]:,} in / {stats[3]:,} out (Cap: {cfg.max_allowed_tokens:,}).")


def interactive_review(cfg: PipelineConfig):
    """CLI reviewer to step through uncertain_words and re-compile volumes."""
    conn = init_db(cfg.output_dir)
    volume_groups = resolve_volumes(cfg)

    for vol_title, _, pdf_list in volume_groups:
        for pdf_path in pdf_list:
            journal_name = pdf_path.stem
            json_dir = cfg.output_dir / journal_name
            img_dir = cfg.clean_img_dir / journal_name
            if not json_dir.exists():
                continue

            for json_file in sorted(json_dir.glob("page_*.json")):
                data = PageTranscription.model_validate_json(json_file.read_text(encoding="utf-8"))
                if not data.uncertain_words:
                    continue

                page_id = json_file.stem
                img_path = img_dir / f"{page_id}.png"
                print(f"\n=== [{vol_title}] {journal_name} -> {page_id} ({len(data.uncertain_words)} items) ===")
                if img_path.exists():
                    if sys.platform.startswith("win"):
                        os.startfile(img_path)
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(img_path)])
                    else:
                        subprocess.Popen(["xdg-open", str(img_path)])

                remaining, modified = [], False
                for idx, item in enumerate(data.uncertain_words, start=1):
                    print(f"\n[{idx}/{len(data.uncertain_words)}] Context: \"{item.line_context}\"")
                    print(f"  LLM Guess : {item.guess}")
                    for alt_idx, alt in enumerate(item.alternatives, start=1):
                        print(f"  Option [{alt_idx}]: {alt}")

                    choice = input("Action -> [Enter]=Accept | [1-9]=Option | [s]=Skip | Or type word: ").strip()
                    if choice.lower() == "s":
                        remaining.append(item)
                        continue
                    replacement = (
                        item.guess if choice == ""
                        else item.alternatives[int(choice) - 1] if (choice.isdigit() and 1 <= int(choice) <= len(item.alternatives))
                        else choice
                    )
                    target = f"[{item.guess}?]"
                    data.transcription = data.transcription.replace(target if target in data.transcription else item.guess, replacement, 1)
                    modified = True

                if modified:
                    data.uncertain_words = remaining
                    atomic_write_text(json_file, data.model_dump_json(indent=2))

    compile_all_volumes(cfg, volume_groups, conn)
    conn.close()


# ==========================================
# 6. Main Pipeline Execution
# ==========================================

def run_pipeline(cfg: PipelineConfig):
    cfg.clean_img_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    conn = init_db(cfg.output_dir)
    client = genai.Client()
    volume_groups = resolve_volumes(cfg)

    if not volume_groups:
        print(f"No PDFs found in {cfg.input_dir.resolve()}.")
        return

    for vol_title, _, pdf_list in volume_groups:
        # Reset sentence continuity at the start of each new Volume,
        # but carry it seamlessly across all PDFs inside the same Volume.
        prev_page_tail = ""

        for pdf_path in pdf_list:
            journal_name = pdf_path.stem
            journal_img_dir = cfg.clean_img_dir / journal_name
            journal_out_dir = cfg.output_dir / journal_name
            journal_img_dir.mkdir(exist_ok=True)
            journal_out_dir.mkdir(exist_ok=True)

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            print(f"\n=== [{vol_title}] {journal_name} ({total_pages} pages) ===")

            try:
                for idx in range(1, total_pages + 1):
                    page_id = f"page_{idx:03d}"
                    img_out_path = journal_img_dir / f"{page_id}.png"
                    json_out_path = journal_out_dir / f"{page_id}.json"

                    row = conn.execute(
                        "SELECT image_cleaned, transcribed, tail_context FROM page_progress WHERE journal_name=? AND page_num=?",
                        (journal_name, idx)
                    ).fetchone()

                    img_done = bool(row[0]) if row else False
                    tx_done = bool(row[1]) if row else False

                    # Step A: Lazy Image Extraction & Cleanup
                    if not img_done or not img_out_path.exists():
                        raw_img = render_single_page(doc, page_index=idx - 1, dpi=cfg.dpi)
                        cleaned_img = clean_photocopy(raw_img)
                        atomic_write_image(img_out_path, cleaned_img)

                        conn.execute("""
                            INSERT INTO page_progress (journal_name, page_num, page_id, image_cleaned)
                            VALUES (?, ?, ?, 1)
                            ON CONFLICT(journal_name, page_num) DO UPDATE SET image_cleaned=1, updated_at=CURRENT_TIMESTAMP
                        """, (journal_name, idx, page_id))
                        conn.commit()
                        print(f"  [{idx}/{total_pages}] Cleaned & saved {page_id}.png")

                    # Step B: LLM Transcription
                    if tx_done and json_out_path.exists():
                        prev_page_tail = row[2] or ""
                        print(f"  [{idx}/{total_pages}] Skipping {page_id} (Already transcribed)")
                        continue

                    # Step C: Hard Cost Circuit Breaker
                    tot_in, tot_out = conn.execute(
                        "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM page_progress"
                    ).fetchone()
                    if (tot_in + tot_out) >= cfg.max_allowed_tokens:
                        raise QuotaExhaustedException(
                            f"Safety brake triggered: {tot_in + tot_out:,} tokens used (limit: {cfg.max_allowed_tokens:,})."
                        )

                    print(f"  [{idx}/{total_pages}] Transcribing {page_id}...")
                    result, in_tok, out_tok = transcribe_page_with_retry(
                        client, cfg, img_out_path, prev_context=prev_page_tail
                    )

                    atomic_write_text(json_out_path, result.model_dump_json(indent=2))
                    prev_page_tail = result.transcription[-200:] if result.transcription else ""

                    conn.execute("""
                        UPDATE page_progress 
                        SET transcribed=1, prompt_tokens=?, output_tokens=?, tail_context=?, updated_at=CURRENT_TIMESTAMP
                        WHERE journal_name=? AND page_num=?
                    """, (in_tok, out_tok, prev_page_tail, journal_name, idx))
                    conn.commit()
                    print(f"    -> Saved ({in_tok} in / {out_tok} out tokens)")

            except (QuotaExhaustedException, KeyboardInterrupt) as stop_err:
                print(f"\n[PAUSED] {stop_err}")
                print("Compiling volume progress so far before exiting...")
                doc.close()
                compile_all_volumes(cfg, volume_groups, conn)
                conn.close()
                sys.exit(0)

            doc.close()

    compile_all_volumes(cfg, volume_groups, conn)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and transcribe scanned/photocopied PDF journals.")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.json"), help="Path to config.json")
    parser.add_argument("--review", action="store_true", help="Launch interactive proofreading CLI for uncertain words")
    parser.add_argument("--compile-only", action="store_true", help="Rebuild Volume Markdown files from existing JSONs without calling API")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.review:
        interactive_review(config)
    elif args.compile_only:
        db_conn = init_db(config.output_dir)
        compile_all_volumes(config, resolve_volumes(config), db_conn)
        db_conn.close()
    else:
        run_pipeline(config)