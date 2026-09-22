import { rm } from "node:fs/promises";

await Promise.all([
  rm(new URL("../dist", import.meta.url), { force: true, recursive: true }),
  rm(new URL("../_wasm", import.meta.url), { force: true, recursive: true }),
  rm(new URL("../src/_package-version.ts", import.meta.url), { force: true }),
]);
