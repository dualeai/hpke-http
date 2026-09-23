import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { delimiter, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(fileURLToPath(new URL("../", import.meta.url)));
const chrome = await findChrome();
let resolveBrowserResult = () => {};
let pendingRelay;
const browserResult = new Promise((resolveResult) => {
  resolveBrowserResult = resolveResult;
});
const server = createServer((request, response) => {
  void serve(request, response).catch((error) => {
    if (response.headersSent) {
      response.destroy(error);
    } else {
      response.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      response.end(String(error));
    }
  });
});

await new Promise((resolveListen, rejectListen) => {
  server.once("error", rejectListen);
  server.listen(0, "127.0.0.1", resolveListen);
});

const address = server.address();
if (address === null || typeof address === "string") {
  throw new Error("browser smoke server did not expose a TCP address");
}

const profile = await mkdtemp(join(tmpdir(), "hpke-http-browser-"));
const url = `http://127.0.0.1:${address.port}/test/browser-smoke.html`;
let execution;
try {
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
  execution = run(chrome, args);
  const outcome = await withTimeout(
    Promise.race([
      browserResult.then((result) => ({ kind: "browser", result })),
      execution.completion.then((result) => ({ kind: "process", result })),
    ]),
    30_000,
    "browser smoke test timed out",
  );
  if (outcome.kind === "process") {
    throw new Error(
      [
        `browser exited before reporting a result (code ${outcome.result.code}, signal ${outcome.result.signal})`,
        outcome.result.error?.stack ?? "",
        outcome.result.stdout,
        outcome.result.stderr,
      ].join("\n"),
    );
  }
  if (outcome.result.status !== "pass") {
    throw new Error(`browser smoke test failed: ${outcome.result.detail}`);
  }
} finally {
  if (execution !== undefined) {
    await stop(execution);
  }
  await new Promise((resolveClose, rejectClose) => {
    server.close((error) =>
      error === undefined ? resolveClose() : rejectClose(error),
    );
  });
  await rm(profile, { recursive: true, force: true });
}

async function serve(request, response) {
  const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
  const pathname = decodeURIComponent(requestUrl.pathname);
  if (pathname === "/relay") {
    if (request.method === "PUT") {
      if (pendingRelay === undefined) {
        response.writeHead(409).end();
        return;
      }
      const held = pendingRelay;
      if (held.middle === undefined) {
        pendingRelay = undefined;
        held.response.end(held.last);
      } else {
        held.response.write(held.middle);
        held.middle = undefined;
      }
      response.writeHead(204).end();
      return;
    }
    if (request.method !== "POST" || pendingRelay !== undefined) {
      response.writeHead(409).end();
      return;
    }
    let body = "";
    for await (const part of request) {
      body += part;
      if (body.length > 65_536) { throw new Error("browser relay input exceeds 64 KiB"); }
    }
    const parts = JSON.parse(body);
    if (!Array.isArray(parts.first) || !Array.isArray(parts.last) ||
        (parts.middle !== undefined && !Array.isArray(parts.middle))) {
      throw new Error("browser relay needs byte arrays");
    }
    const first = Buffer.from(parts.first);
    const middle = parts.middle === undefined ? undefined : Buffer.from(parts.middle);
    const last = Buffer.from(parts.last);
    response.writeHead(200, { "content-type": "message/hpke-http-response", "cache-control": "no-store" });
    response.write(first);
    pendingRelay = { response, middle, last };
    response.once("close", () => {
      if (pendingRelay?.response === response) { pendingRelay = undefined; }
    });
    return;
  }
  if (pathname === "/result") {
    if (request.method !== "POST") {
      response.writeHead(405).end();
      return;
    }
    request.setEncoding("utf8");
    let detail = "";
    for await (const chunk of request) {
      detail += chunk;
      if (detail.length > 65_536) {
        throw new Error("browser result exceeds 64 KiB");
      }
    }
    response.writeHead(204).end();
    resolveBrowserResult({ status: requestUrl.searchParams.get("status"), detail });
    return;
  }
  const allowed =
    pathname === "/test/browser-smoke.html" ||
    pathname.startsWith("/dist/") ||
    pathname.startsWith("/_wasm/browser/");
  const path = resolve(packageRoot, `.${pathname}`);
  if (!allowed || (path !== packageRoot && !path.startsWith(`${packageRoot}${sep}`))) {
    response.writeHead(404).end();
    return;
  }
  let body;
  try {
    body = await readFile(path);
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      response.writeHead(404).end();
      return;
    }
    throw error;
  }
  const contentType = new Map([
    [".html", "text/html; charset=utf-8"],
    [".js", "text/javascript; charset=utf-8"],
    [".wasm", "application/wasm"],
  ]).get(extname(path));
  response.writeHead(200, {
    "cache-control": "no-store",
    "content-length": body.byteLength,
    "content-type": contentType ?? "application/octet-stream",
  });
  response.end(body);
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
    process.env.PROGRAMFILES &&
      join(process.env.PROGRAMFILES, "Google/Chrome/Application/chrome.exe"),
    process.env["PROGRAMFILES(X86)"] &&
      join(process.env["PROGRAMFILES(X86)"], "Google/Chrome/Application/chrome.exe"),
  ].filter(Boolean);
  for (const directory of (process.env.PATH ?? "").split(delimiter)) {
    if (directory) {
      candidates.push(
        ...["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"].map(
          (name) => join(directory, name),
        ),
      );
    }
  }
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Continue to the next conventional installation path.
    }
  }
  throw new Error(
    "Chrome or Chromium is required for the browser smoke test; set CHROME_BIN explicitly",
  );
}

function run(command, args) {
  const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  const completion = new Promise((resolveRun) => {
    child.once("error", (error) => {
      resolveRun({ code: null, signal: null, error, stdout, stderr });
    });
    child.once("close", (code, signal) => {
      resolveRun({ code, signal, error: undefined, stdout, stderr });
    });
  });
  return { child, completion };
}

async function stop(execution) {
  if (execution.child.exitCode !== null || execution.child.signalCode !== null) {
    return;
  }
  execution.child.kill("SIGTERM");
  try {
    await withTimeout(execution.completion, 5_000, "browser did not stop after SIGTERM");
  } catch {
    execution.child.kill("SIGKILL");
    await execution.completion;
  }
}

async function withTimeout(promise, milliseconds, message) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error(message)), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}
