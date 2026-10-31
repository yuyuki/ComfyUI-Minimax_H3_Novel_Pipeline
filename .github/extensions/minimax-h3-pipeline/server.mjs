import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { once } from "node:events";
import { categories, pipeline } from "./catalog.mjs";

const assets = new Map([
  ["/", { body: await readFile(new URL("./index.html", import.meta.url)), type: "text/html; charset=utf-8" }],
  ["/app.js", { body: await readFile(new URL("./app.js", import.meta.url)), type: "text/javascript; charset=utf-8" }],
  ["/styles.css", { body: await readFile(new URL("./styles.css", import.meta.url)), type: "text/css; charset=utf-8" }],
]);

const securityHeaders = {
  "Cache-Control": "no-store",
  "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; object-src 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};

function respond(response, status, contentType, body) {
  response.writeHead(status, {
    ...securityHeaders,
    "Content-Type": contentType,
  });
  response.end(body);
}

export async function startPipelineServer() {
  const server = createServer((request, response) => {
    if (request.method !== "GET") {
      response.setHeader("Allow", "GET");
      respond(response, 405, "text/plain; charset=utf-8", "Method not allowed");
      return;
    }

    let pathname;
    try {
      pathname = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
    } catch {
      respond(response, 400, "text/plain; charset=utf-8", "Invalid request URL");
      return;
    }

    if (pathname === "/api/catalog") {
      respond(
        response,
        200,
        "application/json; charset=utf-8",
        JSON.stringify({ ...pipeline, categories }),
      );
      return;
    }

    const asset = assets.get(pathname);
    if (!asset) {
      respond(response, 404, "text/plain; charset=utf-8", "Not found");
      return;
    }

    respond(response, 200, asset.type, asset.body);
  });

  await once(server.listen(0, "127.0.0.1"), "listening");
  const address = server.address();
  if (!address || typeof address === "string") {
    await new Promise((resolve) => server.close(resolve));
    throw new Error("The canvas server did not receive a TCP address.");
  }

  return {
    url: `http://127.0.0.1:${address.port}/`,
    close: () =>
      new Promise((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
            return;
          }
          resolve();
        });
      }),
  };
}
