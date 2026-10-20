"""Extraction of raw ingredient proximate and available carbohydrate values from IFCT 2017 PDF.

This script extracts Table 1 (Proximate Principles and Dietary Fibre) and Table 6
(Starch and Individual Sugars) for ingredients used in the Indian meal library.
All values are extracted directly from the local PDF and never from memory.
Raw extracts and the book PDF are strictly kept out of version control.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf


def clean_num(s: str) -> float:
    """Parse numeric values from strings containing plus-minus standard deviations."""
    if not s:
        return 0.0
    clean = re.split(r"[±\s]", s.strip())[0]
    try:
        return float(clean)
    except ValueError:
        return 0.0


def extract_table1_rows(doc: pymupdf.Document) -> Dict[str, Dict[str, Any]]:
    """Extract all Table 1 (Proximate Principles) rows across PDF pages 41 to 68."""
    t1_rows: Dict[str, Dict[str, Any]] = {}
    for p in range(40, 68):
        page = doc[p]
        words = page.get_text("words")
        code_words = [w for w in words if re.match(r"^[A-S]\d{3}$", w[4])]
        for cw in code_words:
            code = cw[4]
            y0 = cw[1]
            row = [w for w in words if abs(w[1] - y0) < 4.0]
            row.sort(key=lambda w: w[0])

            is_animal = code[0] in ["N", "O", "P", "Q", "R", "S"] or (
                code[0] == "M" and code in ["M004", "M005", "M006", "M007", "M008", "M009"]
            )

            name_words = [w[4] for w in row if 60 <= w[0] < 270 and w[4] != code]
            name = " ".join(name_words)

            vals: Dict[str, float] = {}
            for w in row:
                x = w[0]
                val = clean_num(w[4])
                if is_animal:
                    if 320 <= x < 390:
                        vals["water"] = val
                    elif 390 <= x < 480:
                        vals["protein"] = val
                    elif 480 <= x < 570:
                        vals["ash"] = val
                    elif 570 <= x < 660:
                        vals["fat"] = val
                    elif 660 <= x < 760:
                        vals["energy_kj"] = val
                    vals["fibre"] = 0.0
                    vals["cho"] = 0.0
                else:
                    if 300 <= x < 355:
                        vals["water"] = val
                    elif 355 <= x < 415:
                        vals["protein"] = val
                    elif 415 <= x < 465:
                        vals["ash"] = val
                    elif 465 <= x < 525:
                        vals["fat"] = val
                    elif 525 <= x < 580:
                        vals["fibre"] = val
                    elif 580 <= x < 635:
                        vals["fibre_ins"] = val
                    elif 635 <= x < 695:
                        vals["fibre_sol"] = val
                    elif 695 <= x < 755:
                        vals["cho"] = val
                    elif 755 <= x < 815:
                        vals["energy_kj"] = val

            if code in ["L001", "L002", "L003", "L004"]:
                vals["fibre"] = 0.0
                for w in row:
                    if 470 <= w[0] < 530:
                        vals["fat"] = clean_num(w[4])
                    if 695 <= w[0] < 740:
                        vals["cho"] = clean_num(w[4])

            raw_text = " ".join(w[4] for w in row)
            t1_rows[code] = {
                "page": p + 1,
                "name": name,
                "raw_text": raw_text,
                "vals": vals,
            }
    return t1_rows


def extract_table6_rows(doc: pymupdf.Document) -> Dict[str, Dict[str, Any]]:
    """Extract all Table 6 (Starch and Sugars) rows across PDF pages 208 to 224."""
    t6_rows: Dict[str, Dict[str, Any]] = {}
    for p in range(207, 224):
        page = doc[p]
        words = page.get_text("words")
        code_words = [w for w in words if re.match(r"^[A-S]\d{3}$", w[4])]
        for cw in code_words:
            code = cw[4]
            y0 = cw[1]
            row = [w for w in words if abs(w[1] - y0) < 4.0]
            row.sort(key=lambda w: w[0])
            vals: Dict[str, float] = {}
            for w in row:
                x = w[0]
                val = clean_num(w[4])
                if 340 <= x < 420:
                    vals["cho_avail"] = val
                elif 420 <= x < 490:
                    vals["starch"] = val
                elif 490 <= x < 560:
                    vals["fructose"] = val
                elif 560 <= x < 625:
                    vals["glucose"] = val
                elif 625 <= x < 690:
                    vals["sucrose"] = val
                elif 690 <= x < 750:
                    vals["maltose"] = val
                elif 750 <= x < 815:
                    vals["free_sugars"] = val

            raw_text = " ".join(w[4] for w in row)
            t6_rows[code] = {
                "page": p + 1,
                "raw_text": raw_text,
                "vals": vals,
            }
    return t6_rows


def load_recipe_codes(recipes_path: Path) -> List[str]:
    """Load distinct IFCT ingredient codes from the recipe assumptions CSV."""
    codes: List[str] = []
    if not recipes_path.exists():
        return codes
    with open(recipes_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = r.get("ifct_code", "").strip()
            if code and code not in codes:
                codes.append(code)
    return codes


def run_extraction(
    pdf_path: Path = Path("data/external/ifct_book/IFCT2017.pdf"),
    recipes_path: Path = Path("data/external/meal_recipes.csv"),
    output_path: Path = Path("data/interim/ifct_extract.csv"),
) -> None:
    """Run full extraction, execute sanity checks, write extract CSV, and print report."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"IFCT 2017 PDF not found at {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    if len(doc) == 0:
        raise ValueError("PDF document is empty or unreadable")

    t1_all = extract_table1_rows(doc)
    t6_all = extract_table6_rows(doc)

    target_codes = load_recipe_codes(recipes_path)
    if not target_codes:
        target_codes = [
            "A010",
            "A011",
            "A013",
            "A015",
            "A018",
            "A019",
            "A022",
            "B001",
            "B002",
            "B003",
            "B010",
            "B021",
            "C033",
            "D046",
            "D075",
            "F002",
            "F006",
            "G017",
            "I001",
            "L001",
            "L002",
            "L003",
            "L004",
            "N003",
            "S006",
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_records: List[Dict[str, Any]] = []
    flagged_records: List[Dict[str, Any]] = []

    print("================================================================================")
    print("IFCT 2017 LOCAL EXTRACTION AND SANITY CHECK REPORT")
    print("================================================================================")
    print(f"Loaded PDF: {pdf_path} ({len(doc)} pages)")
    print(f"Table 1 entries parsed: {len(t1_all)} | Table 6 entries parsed: {len(t6_all)}")
    print("--------------------------------------------------------------------------------")

    for code in target_codes:
        if code not in t1_all:
            print(
                f"[NOT FOUND] Ingredient code {code} is not in IFCT 2017 Table 1. Leaving it out."
            )
            continue

        t1_item = t1_all[code]
        t6_item = t6_all.get(code)

        page_t1 = t1_item["page"]
        page_t6 = t6_item["page"] if t6_item else None
        name = t1_item["name"]

        prot = t1_item["vals"].get("protein", 0.0)
        fat = t1_item["vals"].get("fat", 0.0)
        fib = t1_item["vals"].get("fibre", 0.0)
        cho_t1 = t1_item["vals"].get("cho", 0.0)
        cho_t6 = t6_item["vals"].get("cho_avail", None) if t6_item else None

        # Consistency rule: Table 1 carbohydrate is by difference (100 - water, protein, fat, ash, fibre).
        # Available carbohydrate from Table 6 cannot exceed it by more than a 2 g/100g analytical margin.
        t6_inconsistent = False
        consistency_reason = ""
        if cho_t6 is not None and (cho_t6 - cho_t1) > 2.0:
            t6_inconsistent = True
            consistency_reason = (
                f"Table 6 available carbohydrate ({cho_t6:.2f} g) exceeds Table 1 carbohydrate by difference "
                f"({cho_t1:.2f} g) by {cho_t6 - cho_t1:.2f} g (>2.0 g/100g margin). "
                f"Treated as table inconsistency; using Table 1 carbohydrate ({cho_t1:.2f} g)."
            )
            cho_used = cho_t1
        elif cho_t6 is not None and cho_t6 > 0:
            cho_used = cho_t6
        else:
            cho_used = cho_t1

        kj = t1_item["vals"].get("energy_kj", 0.0)
        book_kcal = kj / 4.184

        # Dual energy checks: using available carbohydrate and carbohydrate by difference
        calc_kcal_avail = 4.0 * cho_used + 4.0 * prot + 9.0 * fat + 2.0 * fib
        diff_pct_avail = (
            (abs(book_kcal - calc_kcal_avail) / calc_kcal_avail * 100.0)
            if calc_kcal_avail > 0
            else 0.0
        )
        match_avail = diff_pct_avail <= 8.0

        calc_kcal_diff = 4.0 * cho_t1 + 4.0 * prot + 9.0 * fat + 2.0 * fib
        diff_pct_diff = (
            (abs(book_kcal - calc_kcal_diff) / calc_kcal_diff * 100.0)
            if calc_kcal_diff > 0
            else 0.0
        )
        match_diff = diff_pct_diff <= 8.0

        energy_status = "unresolved"
        if match_avail and match_diff:
            energy_status = "matches both"
        elif match_avail:
            energy_status = "matches available"
        elif match_diff:
            energy_status = "matches difference"

        rec = {
            "ifct_code": code,
            "ingredient_name": name,
            "page_table1": page_t1,
            "page_table6": page_t6 or "",
            "protein_g": round(prot, 2),
            "fat_g": round(fat, 2),
            "fiber_g": round(fib, 2),
            "carbs_table1_g": round(cho_t1, 2),
            "carbs_table6_g": round(cho_t6, 2) if cho_t6 is not None else "",
            "available_carbs_g": round(cho_used, 2),
            "energy_kj": round(kj, 1),
            "energy_kcal_book": round(book_kcal, 1),
            "energy_kcal_calc_avail": round(calc_kcal_avail, 1),
            "diff_pct_avail": round(diff_pct_avail, 2),
            "match_avail": match_avail,
            "energy_kcal_calc_diff": round(calc_kcal_diff, 1),
            "diff_pct_diff": round(diff_pct_diff, 2),
            "match_diff": match_diff,
            "energy_status": energy_status,
            "consistency_flagged": t6_inconsistent,
            "consistency_reason": consistency_reason,
            "raw_text_table1": t1_item["raw_text"],
            "raw_text_table6": t6_item["raw_text"] if t6_item else "",
        }
        extracted_records.append(rec)
        if t6_inconsistent:
            flagged_records.append(rec)

        print(
            f"[{code}] p.{page_t1:02d} | {name[:24]:24s} | "
            f"Book: {book_kcal:5.1f} kcal | "
            f"Avail CHO ({cho_used:5.2f}g): {calc_kcal_avail:5.1f} kcal ({diff_pct_avail:4.1f}%, {'MATCH' if match_avail else 'NO'}) | "
            f"Diff CHO ({cho_t1:5.2f}g): {calc_kcal_diff:5.1f} kcal ({diff_pct_diff:4.1f}%, {'MATCH' if match_diff else 'NO'}) | "
            f"Status: {energy_status}"
        )

    fieldnames = [
        "ifct_code",
        "ingredient_name",
        "page_table1",
        "page_table6",
        "protein_g",
        "fat_g",
        "fiber_g",
        "carbs_table1_g",
        "carbs_table6_g",
        "available_carbs_g",
        "energy_kj",
        "energy_kcal_book",
        "energy_kcal_calc_avail",
        "diff_pct_avail",
        "match_avail",
        "energy_kcal_calc_diff",
        "diff_pct_diff",
        "match_diff",
        "energy_status",
        "consistency_flagged",
        "consistency_reason",
        "raw_text_table1",
        "raw_text_table6",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_records)

    print("--------------------------------------------------------------------------------")
    print(f"Wrote {len(extracted_records)} extracted ingredient rows to {output_path}")
    print(f"Total consistency flagged rows: {len(flagged_records)} / {len(extracted_records)}")
    if flagged_records:
        print("\nConsistency Flagged Ingredients Breakdown:")
        for fr in flagged_records:
            print(
                f"  - [{fr['ifct_code']}] {fr['ingredient_name']}: "
                f"Table 6={fr['carbs_table6_g']} g vs Table 1={fr['carbs_table1_g']} g. "
                f"Reason: {fr['consistency_reason']}"
            )
    print("================================================================================")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_extraction()
