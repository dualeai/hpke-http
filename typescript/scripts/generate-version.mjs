import { readFile, writeFile } from "node:fs/promises";

const packageUrl = new URL("../package.json", import.meta.url);
const outputUrl = new URL("../src/_package-version.ts", import.meta.url);
const metadata = JSON.parse(await readFile(packageUrl, "utf8"));

if (typeof metadata.version !== "string" || metadata.version.length === 0) {
  throw new Error("typescript/package.json must contain a non-empty version string");
}

await writeFile(
  outputUrl,
  `// Generated from package.json. Do not edit.\nexport const PACKAGE_VERSION = ${JSON.stringify(metadata.version)} as const;\n`,
  "utf8",
);
