import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { delimiter, extname, join, resolve, sep } from "node:path";

import { build, version as viteVersion } from "vite8";

assert.equal(viteVersion, "8.2.2", "the artifact smoke test requires Vite 8.2.2");
assert.equal(process.argv.length, 3, "pass exactly one npm tarball path");

const archive = resolve(process.argv[2]);
await access(archive, constants.R_OK);

const fixture = await mkdtemp(join(tmpdir(), "hpke-http-vite-smoke-"));
const output = join(fixture, "dist");
const installedWasm = join(
  fixture,
  "node_modules/@dualeai/hpke-http/_wasm/browser/hpke_http_wasm_bg.wasm",
);
let server;
let pendingResult;
let explicitRequests = 0;

try {
  await writeFile(
    join(fixture, "package.json"),
    JSON.stringify({ name: "hpke-http-vite-smoke", private: true, type: "module" }),
  );
  execFileSync(
    "npm",
    ["install", "--ignore-scripts", "--no-audit", "--no-fund", archive],
    { cwd: fixture, encoding: "utf8", stdio: "pipe" },
  );
  await writeFile(
    join(fixture, "index.html"),
    '<!doctype html><meta charset="utf-8"><script type="module" src="./main.js"></script>\n',
  );
  await writeFile(
    join(fixture, "main.js"),
    [
      'import { initialize, isInitialized, generateKeyPair } from "@dualeai/hpke-http/browser";',
      'const mode = new URL(location.href).searchParams.get("mode");',
      'async function report(status, detail = "") {',
      '  await fetch("/result?mode=" + encodeURIComponent(mode) + "&status=" + status, {',
      '    method: "POST", body: detail,',
      '  });',
      '}',
      'try {',
      '  if (mode === "explicit") {',
      '    await initialize(new URL("/explicit.wasm", location.href));',
      '  } else if (mode === "default") {',
      '    await initialize();',
      '  } else {',
      '    throw new Error("unknown browser mode");',
      '  }',
      '  if (!isInitialized()) throw new Error("browser WASM did not initialize");',
      '  const keys = generateKeyPair();',
      '  if (keys.publicKey.byteLength !== 32 || keys.privateKey.byteLength !== 32) {',
      '    throw new Error("browser WASM returned invalid keys");',
      '  }',
      '  keys.privateKey.fill(0);',
      '  await report("pass");',
      '} catch (error) {',
      '  await report("fail", String(error?.stack ?? error));',
      '}',
      '',
    ].join("\n"),
  );

  await build({
    root: fixture,
    configFile: false,
    logLevel: "error",
    build: {
      outDir: output,
      emptyOutDir: true,
      manifest: true,
      assetsInlineLimit: 0,
    },
  });

  const manifest = JSON.parse(await readFile(join(output, ".vite/manifest.json"), "utf8"));
  const entry = manifest["index.html"];
  assert.ok(entry?.file, "Vite manifest has no HTML entry");
  const glueKey = entry.dynamicImports?.find((key) =>
    key.endsWith("/_wasm/browser/hpke_http_wasm.js"),
  );
  assert.ok(glueKey, "Vite did not emit the browser WASM glue import");
  const glue = manifest[glueKey];
  assert.ok(glue?.file?.endsWith(".js"), "Vite manifest has no browser glue chunk");
  const wasm = glue.assets?.find((file) => file.endsWith(".wasm"));
  assert.ok(wasm, "Vite manifest has no WASM asset linked to the glue");
  await Promise.all([access(join(output, glue.file)), access(join(output, wasm))]);

  server = createServer((request, response) => {
    void serve(request, response).catch((error) => {
      response.writeHead(500, { "content-type": "text/plain" }).end(String(error));
    });
  });
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  assert.ok(address && typeof address !== "string");

  const chrome = await findChrome();
  await runBrowser(chrome, address.port, "default");
  await runBrowser(chrome, address.port, "explicit");
  assert.ok(explicitRequests > 0, "the browser did not request explicit.wasm");
  process.stdout.write("packed Vite 8 browser smoke passed\n");
} finally {
  if (server !== undefined) {
    await new Promise((resolveClose) => server.close(resolveClose));
  }
  await rm(fixture, { recursive: true, force: true });
}

async function serve(request, response) {
  const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
  const pathname = decodeURIComponent(requestUrl.pathname);
  if (pathname === "/result") {
    if (request.method !== "POST") {
      response.writeHead(405).end();
      return;
    }
    let detail = "";
    for await (const part of request) {
      detail += part;
    }
    response.writeHead(204).end();
    if (pendingResult?.mode === requestUrl.searchParams.get("mode")) {
      pendingResult.resolve({
        status: requestUrl.searchParams.get("status"),
        detail,
      });
    }
    return;
  }
  if (pathname === "/explicit.wasm") {
    explicitRequests += 1;
    const bytes = await readFile(installedWasm);
    response.writeHead(200, {
      "content-type": "application/wasm",
      "cache-control": "no-store",
    }).end(bytes);
    return;
  }
  const file = resolve(output, `.${pathname === "/" ? "/index.html" : pathname}`);
  if (!file.startsWith(`${output}${sep}`)) {
    response.writeHead(404).end();
    return;
  }
  let bytes;
  try {
    bytes = await readFile(file);
  } catch (error) {
    if (error?.code === "ENOENT") {
      response.writeHead(404).end();
      return;
    }
    throw error;
  }
  const mediaType = new Map([
    [".html", "text/html; charset=utf-8"],
    [".js", "text/javascript; charset=utf-8"],
    [".wasm", "application/wasm"],
  ]).get(extname(file)) ?? "application/octet-stream";
  response.writeHead(200, {
    "content-type": mediaType,
    "cache-control": "no-store",
  }).end(bytes);
}

async function runBrowser(chrome, port, mode) {
  const profile = await mkdtemp(join(tmpdir(), "hpke-http-vite-chrome-"));
  const result = new Promise((resolveResult) => {
    pendingResult = { mode, resolve: resolveResult };
  });
  const url = `http://127.0.0.1:${port}/?mode=${mode}`;
  const args = [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-default-browser-check",
    "--no-first-run",
    `--user-data-dir=${profile}`,
    url,
  ];
  if (typeof process.getuid === "function" && process.getuid() === 0) {
    args.unshift("--no-sandbox");
  }
  const child = spawn(chrome, args, { stdio: ["ignore", "pipe", "pipe"] });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (part) => { stderr += part; });
  child.stdout.resume();
  const exit = new Promise((resolveExit) => {
    child.once("error", (error) => resolveExit({ error }));
    child.once("close", (code, signal) => resolveExit({ code, signal }));
  });
  try {
    const outcome = await withTimeout(
      Promise.race([
        result.then((value) => ({ kind: "result", value })),
        exit.then((value) => ({ kind: "exit", value })),
      ]),
      30_000,
      `Chrome timed out during ${mode} initialization`,
    );
    if (outcome.kind === "exit") {
      throw new Error(`Chrome exited during ${mode} initialization: ${JSON.stringify(outcome.value)}\n${stderr}`);
    }
    if (outcome.value.status !== "pass") {
      throw new Error(`${mode} initialization failed: ${outcome.value.detail}\n${stderr}`);
    }
  } finally {
    pendingResult = undefined;
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
    }
    try {
      await withTimeout(exit, 5_000, "Chrome did not stop");
    } catch {
      child.kill("SIGKILL");
      await exit;
    }
    await rm(profile, { recursive: true, force: true });
  }
}

async function findChrome() {
  const candidates = [
    process.env.CHROME_BIN,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  for (const directory of (process.env.PATH ?? "").split(delimiter)) {
    for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
      candidates.push(join(directory, name));
    }
  }
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Try the next Chrome path.
    }
  }
  throw new Error("Chrome or Chromium is required; set CHROME_BIN");
}

async function withTimeout(promise, milliseconds, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
