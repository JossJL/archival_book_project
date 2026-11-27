import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

# Import the shared schemas, config loader, and volume compiler from your main script
from transcribe_typewritten_script import (
    load_config,
    resolve_volumes,
    init_db,
    atomic_write_text,
    compile_all_volumes,
    PageTranscription,
    PipelineConfig,
)


def open_image(img_path: Path):
    """Opens the cleaned page image in the OS default viewer."""
    if not img_path.exists():
        print(f"  [!] Image not found at: {img_path}")
        return
    if sys.platform.startswith("win"):
        os.startfile(img_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(img_path)])
    else:
        subprocess.Popen(["xdg-open", str(img_path)])


def replace_uncertain_token(text: str, guess: str, replacement: str) -> tuple[str, bool]:
    """
    Safely replaces [guess?] or [guess] in the transcription text using regex.
    Returns (updated_text, was_replaced).
    """
    # 1. Match bracketed forms first: [guess?], [guess ?], or [guess] (case-insensitive)
    bracket_pattern = re.compile(r"\[\s*" + re.escape(guess) + r"\s*\??\s*\]", re.IGNORECASE)
    if bracket_pattern.search(text):
        return bracket_pattern.sub(replacement, text, count=1), True

    # 2. Fallback: match exact whole word if the LLM forgot brackets in the body text
    word_pattern = re.compile(r"\b" + re.escape(guess) + r"\b")
    if word_pattern.search(text):
        return word_pattern.sub(replacement, text, count=1), True

    return text, False


def interactive_review(cfg: PipelineConfig, review_notes: bool = False, include_notes_in_book: bool = False):
    conn = init_db(cfg.output_dir)
    volume_groups = resolve_volumes(cfg)

    print("\n========================================================")
    print(" INTERACTIVE EDITORIAL REVIEW")
    print(" Every decision is saved immediately. Press Ctrl+C anytime to pause.")
    print("========================================================")

    try:
        for vol_title, _, pdf_list in volume_groups:
            for pdf_path in pdf_list:
                journal_name = pdf_path.stem
                journal_dir = cfg.output_dir / journal_name
                img_dir = cfg.clean_img_dir / journal_name

                if not journal_dir.exists():
                    continue

                for json_file in sorted(journal_dir.glob("page_*.json")):
                    data = PageTranscription.model_validate_json(json_file.read_text(encoding="utf-8"))
                    
                    has_words = bool(data.uncertain_words)
                    has_notes = review_notes and bool(data.archivist_notes)

                    if not has_words and not has_notes:
                        continue

                    page_id = json_file.stem
                    img_path = img_dir / f"{page_id}.png"
                    
                    print(f"\n==========================================")
                    print(f" [{vol_title}] {journal_name} -> {page_id}")
                    print(f" Image: {img_path}")
                    print(f"==========================================")
                    open_image(img_path)

                    # ------------------------------------------
                    # Part 1: Resolve Uncertain Words
                    # ------------------------------------------
                    if has_words:
                        remaining = list(data.uncertain_words)
                        total_items = len(data.uncertain_words)

                        for idx, item in enumerate(list(data.uncertain_words), start=1):
                            print(f"\n[{idx}/{total_items}] Context: \"{item.line_context}\"")
                            print(f"  LLM Guess   : {item.guess}")
                            for alt_idx, alt in enumerate(item.alternatives, start=1):
                                print(f"  Option [{alt_idx}]  : {alt}")

                            print("\n  [Enter] Accept Guess (strip brackets)")
                            print("  [1-9]   Pick Option #")
                            print("  [k]     Keep as [guess?] in book, but clear from review queue")
                            print("  [i]     Mark as *[illegible]*")
                            print("  [s]     Skip for later")
                            choice = input("  Action or type custom replacement: ").strip()

                            if choice.lower() == "s":
                                continue
                            elif choice.lower() == "k":
                                replacement = f"[{item.guess}?]"
                            elif choice.lower() == "i":
                                replacement = "*[illegible]*"
                            elif choice == "":
                                replacement = item.guess
                            elif choice.isdigit() and 1 <= int(choice) <= len(item.alternatives):
                                replacement = item.alternatives[int(choice) - 1]
                            else:
                                replacement = choice

                            new_text, ok = replace_uncertain_token(data.transcription, item.guess, replacement)
                            if ok:
                                data.transcription = new_text
                                print(f"  -> Applied: '{replacement}'")
                            else:
                                print(f"  [!] Could not auto-locate '{item.guess}' in text. Check {json_file.name} manually.")
                                continue

                            # Save immediately after each resolved word
                            remaining.remove(item)
                            data.uncertain_words = remaining
                            atomic_write_text(json_file, data.model_dump_json(indent=2))

                    # ------------------------------------------
                    # Part 2: Optional Archivist Note Triage
                    # ------------------------------------------
                    if review_notes and data.archivist_notes:
                        print(f"\nArchivist Note on {page_id}:")
                        print(f"  \"{data.archivist_notes}\"")
                        note_choice = input(
                            "  Action -> [Enter]=Keep Note | [d]=Delete Note | Or type custom family footnote: "
                        ).strip()
                        
                        if note_choice.lower() == "d":
                            data.archivist_notes = None
                            atomic_write_text(json_file, data.model_dump_json(indent=2))
                            print("  -> Note deleted.")
                        elif note_choice != "":
                            data.archivist_notes = note_choice
                            atomic_write_text(json_file, data.model_dump_json(indent=2))
                            print("  -> Note updated.")

    except KeyboardInterrupt:
        print("\n\n[Paused] Saving progress and re-compiling Volume Markdown files...")

    # Re-compile Volume_1 and Volume_2 Markdown files with all updates
    compile_all_volumes(cfg, volume_groups, conn, include_notes=include_notes_in_book)
    conn.close()
    print("\nReview session complete! Volume Markdown files updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive proofreading tool for uncertain words and notes.")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.json"), help="Path to config.json")
    parser.add_argument("--review-notes", action="store_true", help="Also review/delete AI archivist notes on each page")
    parser.add_argument("--with-notes", action="store_true", help="Include remaining archivist_notes in the compiled book Markdown")
    args = parser.parse_args()

    config = load_config(args.config)
    interactive_review(
        config,
        review_notes=args.review_notes,
        include_notes_in_book=args.with_notes
    )