/**
 * Copies source/lib to out/html/lib and verifies the copy succeeded.
 *
 * WHY THIS EXISTS: out/html/lib is gitignored (it's a generated artifact). The old
 * shell links to lib/index.js in index.html. A silent copy means an invisible dead
 * shell — the browser loads a stale build or no script at all with zero error signal,
 * just a quietly broken simulation (LEARNINGS 2026-07-19). This tool exits loudly
 * if the copy fails, and prints every file so the copy is never silent.
 *
 * Usage: node tools/copy-game-lib.mjs   (exit 0 on success, 1 on failure)
 */
import { cpSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const sourcePath = join(root, 'source', 'lib');
const targetPath = join(root, 'out', 'html', 'lib');

// Copy the entire directory recursively
cpSync(sourcePath, targetPath, { recursive: true, force: true });

// Verify the critical file exists
const indexPath = join(targetPath, 'index.js');
if (!existsSync(indexPath)) {
  console.error(`✘ copy-game-lib FAILED: ${indexPath} does not exist after copy`);
  process.exit(1);
}

// Walk the copied directory and print every file
function walkDir(dirPath, basePath) {
  const entries = readdirSync(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name);
    const relativePath = fullPath.slice(basePath.length + 1); // +1 to skip the separator
    if (entry.isFile()) {
      console.log(`  ${relativePath}`);
    } else if (entry.isDirectory()) {
      walkDir(fullPath, basePath);
    }
  }
}

console.log('Copied source/lib → out/html/lib:');
walkDir(targetPath, targetPath);
console.log(`✓ copy-game-lib: ${indexPath} verified`);
