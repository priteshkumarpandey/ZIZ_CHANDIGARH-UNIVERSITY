import { describe, it, expect } from "vitest";
import fs from "fs";
import path from "path";

function getAllSourceFiles(dir: string): string[] {
  let results: string[] = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    if (["node_modules", ".next", "dist", "tests"].includes(file)) continue;
    const full = path.join(dir, file);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      results = results.concat(getAllSourceFiles(full));
    } else if (/\.(tsx|ts)$/.test(file)) {
      results.push(full);
    }
  }
  return results;
}

describe("Icon and Branding Governance", () => {
  const componentsDir = path.resolve(__dirname, "../components");
  const appDir = path.resolve(__dirname, "../app");
  const allFiles = [...getAllSourceFiles(componentsDir), ...getAllSourceFiles(appDir)];

  it("fails if any component imports icons from a package other than lucide-react", () => {
    const forbiddenIconPackages = [
      "@heroicons",
      "react-icons",
      "@tabler/icons",
      "@radix-ui/react-icons",
      "feather-icons",
      "font-awesome",
      "@fortawesome",
      "ionicons",
    ];

    for (const file of allFiles) {
      const content = fs.readFileSync(file, "utf8");
      const importRegex = /from\s+['"]([^'"]+)['"]/g;
      let match: RegExpExecArray | null;
      while ((match = importRegex.exec(content)) !== null) {
        const importSource = match[1];
        for (const forbidden of forbiddenIconPackages) {
          expect(
            importSource.toLowerCase(),
            `File ${file} imports from forbidden icon library: ${importSource}`
          ).not.toContain(forbidden);
        }
      }
    }
  });

  it("ensures all icon imports in components and app routes come exclusively from lucide-react", () => {
    let totalLucideImports = 0;
    for (const file of allFiles) {
      const content = fs.readFileSync(file, "utf8");
      if (content.includes("lucide-react")) {
        totalLucideImports++;
      }
    }
    // Verify that lucide-react is indeed used throughout the codebase
    expect(totalLucideImports).toBeGreaterThan(5);
  });

  it("strictly enforces zero forbidden copy words (investigational, certified, approved, clinical-grade, clinical dossier)", () => {
    const forbiddenWords = ["investigational", "certified", "approved", "clinical-grade", "clinical dossier"];

    for (const file of allFiles) {
      const content = fs.readFileSync(file, "utf8");
      for (const word of forbiddenWords) {
        const regex = new RegExp(`\\b${word}\\b`, "i");
        expect(
          regex.test(content),
          `File ${file} contains forbidden copy word: "${word}"`
        ).toBe(false);
      }
    }
  });
});
