import fs from 'fs';
import path from 'path';

const FORBIDDEN_PATTERNS = [
  /shadow-(?:sm|md|lg|xl|2xl)/,
  /drop-shadow-/,
  /text-shadow-/,
  /glow/,
];

function scanDir(dir, errors = []) {
  if (!fs.existsSync(dir)) return errors;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (['node_modules', '.next', '.git'].includes(entry.name)) continue;
      scanDir(fullPath, errors);
    } else if (/\.(tsx|ts|jsx|js|css)$/.test(entry.name)) {
      if (entry.name === 'check_flat_style.mjs') continue;
      const content = fs.readFileSync(fullPath, 'utf8');
      const lines = content.split('\n');
      lines.forEach((line, idx) => {
        for (const pattern of FORBIDDEN_PATTERNS) {
          if (pattern.test(line)) {
            errors.push({
              file: fullPath,
              line: idx + 1,
              match: line.trim(),
              pattern: pattern.toString(),
            });
          }
        }
      });
    }
  }
  return errors;
}

const frontendDir = process.cwd();
const errors = [];
scanDir(path.join(frontendDir, 'app'), errors);
scanDir(path.join(frontendDir, 'components'), errors);

if (errors.length > 0) {
  console.error("Found " + errors.length + " forbidden non-flat style occurrences:");
  for (const err of errors) {
    console.error("  " + err.file + ":" + err.line + " matches " + err.pattern);
    console.error("    -> " + err.match);
  }
  process.exit(1);
} else {
  console.log("Flat design check passed. No forbidden shadows or glows found.");
}

