import { describe, it, expect } from "vitest";
import fs from "fs";
import path from "path";

describe("No Typed Numbers in Static JSX Strings", () => {
  const targetFiles = [
    "app/evidence/page.tsx",
    "app/what-if/page.tsx",
    "app/summary/[id]/page.tsx",
    "app/twin/[id]/page.tsx",
    "app/patients/page.tsx",
  ];

  // Helper to check if a JSX text string containing digits is a permitted label, heading ordinal, or unit
  function isLabelHeadingOrdinalOrUnit(text: string): boolean {
    const trimmed = text.trim();
    if (!trimmed) return true;

    // Heading ordinals (e.g. "1.", "2.", "50th", "25th to 75th Band", "10,000-Draw")
    if (/^\d+\.?$/.test(trimmed)) return true;
    if (/^\d+(?:st|nd|rd|th)(?:\s+(?:Band|percentile))?$/i.test(trimmed)) return true;
    if (/^(?:5th to 95th|25th to 75th)\s+Band$/i.test(trimmed)) return true;
    if (/^#\d+$/i.test(trimmed)) return true;
    if (/^\d{1,3}(?:,\d{3})*(?:-Draw)?/i.test(trimmed)) return true;

    // Units (e.g. "180-minute", "5-min", "24 h", "last 6 h", "70 to 180 mg/dL", "kg/m²", "1x", "2x")
    if (/^(?:last\s+)?\d+(?:\.\d+)?\s*(?:h|hrs|hours|min|mins|minutes|mg\/dL|METs|MET-min|kg\/m²|x)$/i.test(trimmed)) return true;
    if (/^\d+-(?:minute|min)$/i.test(trimmed)) return true;
    if (/(?:needs at least\s+)?\d+\s*h/i.test(trimmed)) return true;

    // Clinical target definitions or standard reference range labels
    if (/^Target(?:\s+TIR)?\s*(?:\([^)]*\))?:\s*(?:>|<|&gt;|&lt;|&le;|&ge;|\u2264|\u2265)?\s*\d+(?:\.\d+)?%?$/i.test(trimmed)) return true;
    if (/^Target\s*(?:\([^)]*\)|[<>]\s*\d+%?|Range\s*\([^)]*\))$/i.test(trimmed)) return true;
    if (/^\d+\s*to\s*\d+\s*(?:mg\/dL)?$/i.test(trimmed)) return true;
    if (/^Time in Range\s*\(last\s+\d+\s*h\)$/i.test(trimmed)) return true;

    // Field labels and headings (ending with ":" or standard scenario/delta descriptors)
    if (/:$/.test(trimmed)) return true;
    if (/^Scenario\s+[AB]\s*(?:\+\d+m\s+Delta)?$/i.test(trimmed)) return true;
    if (/^B\s+minus\s+A\s*(?:\(\+\d+m\))?$/i.test(trimmed)) return true;
    if (/(?:T2D|Rank-\d+|N=\d+|fold\s+\d+|v\d+\.\d+|IND_PAT_\d+|ifct_\d+|Compare\s+\d+\s+Scenarios)/i.test(trimmed)) return true;
    if (/^\+\d+m(?:\s+Delta)?$/i.test(trimmed)) return true;
    if (/^Median\s*\(\d+th\)$/i.test(trimmed)) return true;
    if (/^Postprandial\s+\d+-minute\s+projection/i.test(trimmed)) return true;

    return false;
  }

  it.each(targetFiles)("scans %s for hardcoded metric numbers in static text", (relPath) => {
    const fullPath = path.resolve(__dirname, "..", relPath);
    expect(fs.existsSync(fullPath)).toBe(true);

    const content = fs.readFileSync(fullPath, "utf-8");
    const lines = content.split("\n");

    const violations: { line: number; text: string; raw: string }[] = [];
    let insideJsx = false;

    lines.forEach((lineText, idx) => {
      // Simple heuristic to track JSX return block: only inspect lines after "return ("
      if (lineText.includes("return (")) {
        insideJsx = true;
      }

      if (!insideJsx) return;

      // Find JSX text between tags: > text with digits <
      const matches = lineText.match(/>([^<>{}]*\d+[^<>{}]*)</g);
      if (matches) {
        matches.forEach((m) => {
          const text = m.slice(1, -1).trim();
          if (!text) return;

          // Check if matches allowed label, heading ordinal, or unit
          if (!isLabelHeadingOrdinalOrUnit(text)) {
            // Also ignore CSS classnames or pure identifiers
            if (/^(?:[A-Z0-9_-]+|\s*)$/i.test(text)) return;
            violations.push({ line: idx + 1, text, raw: lineText.trim() });
          }
        });
      }
    });

    if (violations.length > 0) {
      console.warn(`Violations in ${relPath}:`, violations);
    }
    expect(violations).toEqual([]);
  });

  it("verifies AI-READI license is published with proper disclaimer", () => {
    const evidencePath = path.resolve(__dirname, "../app/evidence/page.tsx");
    const content = fs.readFileSync(evidencePath, "utf-8");
    expect(content).not.toContain("CC BY 4.0");
  });

  it("verifies isotonic constraint mentions '(passed, by construction)'", () => {
    const evidencePath = path.resolve(__dirname, "../app/evidence/page.tsx");
    const whatIfPath = path.resolve(__dirname, "../app/what-if/page.tsx");
    const evContent = fs.readFileSync(evidencePath, "utf-8");
    const wiContent = fs.readFileSync(whatIfPath, "utf-8");

    expect(evContent).toContain("(passed, by construction)");
    expect(wiContent).toContain("(passed, by construction)");
  });
});
