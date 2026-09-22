import { mkdir, rename } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const typescriptRoot = fileURLToPath(new URL("..", import.meta.url));
const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const wasmInput = fileURLToPath(
  new URL("../../target/wasm32-unknown-unknown/release/hpke_http_wasm.wasm", import.meta.url),
);
const browserOutput = fileURLToPath(new URL("../_wasm/browser", import.meta.url));
const nodeOutput = fileURLToPath(new URL("../_wasm/node", import.meta.url));

run("wasm-bindgen", ["--version"], typescriptRoot, (output) => {
  if (!output.includes("0.2.128")) {
    throw new Error(`wasm-bindgen-cli 0.2.128 is required; got ${output.trim()}`);
  }
});

run(
  "cargo",
  ["build", "--locked", "--release", "--target", "wasm32-unknown-unknown", "-p", "hpke-http-wasm"],
  repositoryRoot,
);

await Promise.all([
  mkdir(browserOutput, { recursive: true }),
  mkdir(nodeOutput, { recursive: true }),
]);

run(
  "wasm-bindgen",
  ["--target", "web", "--out-dir", browserOutput, "--out-name", "hpke_http_wasm", wasmInput],
  typescriptRoot,
);
run(
  "wasm-bindgen",
  ["--target", "nodejs", "--out-dir", nodeOutput, "--out-name", "hpke_http_wasm", wasmInput],
  typescriptRoot,
);

await rename(`${nodeOutput}/hpke_http_wasm.js`, `${nodeOutput}/hpke_http_wasm.cjs`);

function run(command, arguments_, cwd, inspectOutput) {
  const result = spawnSync(command, arguments_, {
    cwd,
    encoding: "utf8",
    stdio: inspectOutput === undefined ? "inherit" : "pipe",
  });
  if (result.error !== undefined) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} failed with status ${String(result.status)}`);
  }
  if (inspectOutput !== undefined) {
    inspectOutput(`${result.stdout}${result.stderr}`);
  }
}
