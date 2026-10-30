import { describe, expect, it, vi } from "vitest";
import { handleAsk, handleConfig, MAX_QUESTION_CHARS, type Deps, type Env, type LimiterApi } from "../src/ask.ts";
import { createTables, refund, reserve } from "../src/limits.ts";
import { memorySql } from "./sqlite.ts";

const SECRET = "gateway-secret-value-123456";
const env: Env = { ORIGIN_URL: "https://origin.example", GATEWAY_SECRET: SECRET, IP_LIMIT_PER_HOUR: "10", DAILY_CAP: "950" };
const SSE = 'event: retrieval\ndata: {"count":5}\n\nevent: token\ndata: {"text":"Hi"}\n\nevent: done\ndata: {}\n\n';

/** The real limit code on an in-memory database, with the clock under the test's control. */
function realLimiter(cfg = { ipLimitPerHour: 10, dailyCap: 950 }, clock: { now: number } = { now: Date.UTC(2026, 9, 30, 12) }) {
  const sql = memorySql();
  createTables(sql);
  const refunds: number[] = [];
  const api: LimiterApi = {
    reserve: async (ip) => reserve(sql, cfg, ip, clock.now),
    refund: async (id, day) => { refunds.push(id); refund(sql, id, day); },
  };
  return { api, refunds, clock };
}

function sseResponse(body = SSE, init: ResponseInit = {}) {
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" }, ...init });
}

function post(body: unknown, headers: Record<string, string> = {}) {
  return new Request("https://sourcebound.example/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json", "cf-connecting-ip": "203.0.113.7", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

function setup(origin: (req: Request) => Response | Promise<Response> = () => sseResponse(), limiter = realLimiter()) {
  const calls: Request[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = new Request(input as string, init);
    calls.push(req);
    return origin(req);
  });
  const deps: Deps = { limiter: limiter.api, fetch: fetchMock as unknown as typeof fetch };
  return { deps, calls, limiter };
}

describe("forwarding", () => {
  it("sends the question and the gateway secret to the origin, and streams the origin's body through", async () => {
    const { deps, calls } = setup();
    const res = await handleAsk(post({ question: "  What optimizer?  " }), env, deps);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    expect(await res.text()).toBe(SSE);
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://origin.example/ask");
    expect(calls[0].headers.get("x-gateway-secret")).toBe(SECRET);
    expect(await calls[0].json()).toEqual({ question: "What optimizer?" }); // nothing else is forwarded
    expect(calls[0].headers.get("cf-connecting-ip")).toBeNull();
  });

  it("passes chunks through as they arrive instead of buffering the whole answer", async () => {
    let push!: (s: string) => void;
    let close!: () => void;
    const enc = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(c) { push = (s) => c.enqueue(enc.encode(s)); close = () => c.close(); },
    });
    const { deps } = setup(() => new Response(stream, { headers: { "content-type": "text/event-stream" } }));
    const res = await handleAsk(post({ question: "q" }), env, deps);
    const reader = res.body!.getReader();
    push("event: token\ndata: 1\n\n");
    const first = await reader.read();
    expect(new TextDecoder().decode(first.value)).toBe("event: token\ndata: 1\n\n"); // before the origin finished
    close();
    expect((await reader.read()).done).toBe(true);
  });

  it("never puts the secret in a response", async () => {
    for (const origin of [() => sseResponse(), () => new Response("nope", { status: 401 }), () => { throw new Error(`boom ${SECRET}`); }]) {
      const { deps } = setup(origin as never);
      const res = await handleAsk(post({ question: "q" }), env, deps);
      expect(await res.text()).not.toContain(SECRET);
      expect(JSON.stringify([...res.headers])).not.toContain(SECRET);
    }
  });
});

describe("input checks", () => {
  it("rejects a question over 500 characters without touching the origin or the limiter", async () => {
    const { deps, calls } = setup();
    const ok = await handleAsk(post({ question: "q".repeat(MAX_QUESTION_CHARS) }), env, deps);
    expect(ok.status).toBe(200);
    const res = await handleAsk(post({ question: "q".repeat(MAX_QUESTION_CHARS + 1) }), env, deps);
    expect(res.status).toBe(413);
    expect((await res.json()) as { error: string }).toMatchObject({ error: "too_long" });
    expect(calls).toHaveLength(1);
  });

  it("rejects an oversized body, bad JSON, a missing or non-string question, and other methods", async () => {
    const { deps, calls } = setup();
    expect((await handleAsk(post("x".repeat(5000)), env, deps)).status).toBe(413);
    expect((await handleAsk(post("not json"), env, deps)).status).toBe(400);
    expect((await handleAsk(post({}), env, deps)).status).toBe(400);
    expect((await handleAsk(post({ question: 42 }), env, deps)).status).toBe(400);
    expect((await handleAsk(post({ question: "   " }), env, deps)).status).toBe(400);
    expect((await handleAsk(post([]), env, deps)).status).toBe(400);
    const get = await handleAsk(new Request("https://x.example/api/ask"), env, deps);
    expect(get.status).toBe(405);
    expect(calls).toHaveLength(0);
  });

  it("refuses to run without the gateway secret", async () => {
    const { deps, calls } = setup();
    const res = await handleAsk(post({ question: "q" }), { ...env, GATEWAY_SECRET: undefined }, deps);
    expect(res.status).toBe(503);
    expect(calls).toHaveLength(0);
  });
});

describe("limits", () => {
  it("allows 10 questions an hour per client, then answers 429 with Retry-After and no origin call", async () => {
    const { deps, calls } = setup();
    for (let i = 0; i < 10; i++) expect((await handleAsk(post({ question: `q${i}` }), env, deps)).status).toBe(200);
    const res = await handleAsk(post({ question: "one more" }), env, deps);
    expect(res.status).toBe(429);
    const body = (await res.json()) as { error: string; samples: boolean; scope: string };
    expect(body).toMatchObject({ error: "limit_ip", samples: true, scope: "ip" });
    expect(Number(res.headers.get("retry-after"))).toBeGreaterThan(0);
    expect(calls).toHaveLength(10);
    const other = await handleAsk(post({ question: "q" }, { "cf-connecting-ip": "198.51.100.9" }), env, deps);
    expect(other.status).toBe(200);
  });

  it("stops everyone at the daily cap", async () => {
    const lim = realLimiter({ ipLimitPerHour: 10, dailyCap: 3 });
    const { deps, calls } = setup(() => sseResponse(), lim);
    for (let i = 0; i < 3; i++) expect((await handleAsk(post({ question: "q" }, { "cf-connecting-ip": `192.0.2.${i}` }), env, deps)).status).toBe(200);
    const res = await handleAsk(post({ question: "q" }, { "cf-connecting-ip": "192.0.2.99" }), env, deps);
    expect(res.status).toBe(429);
    expect((await res.json()) as object).toMatchObject({ error: "limit_daily", samples: true });
    expect(calls).toHaveLength(3);
  });
});

describe("fallback when the origin cannot answer", () => {
  it("origin unreachable -> 503 origin_unavailable with samples, and the question is not counted", async () => {
    const lim = realLimiter({ ipLimitPerHour: 1, dailyCap: 5 });
    const { deps } = setup(() => { throw new TypeError("fetch failed"); }, lim);
    const res = await handleAsk(post({ question: "q" }), env, deps);
    expect(res.status).toBe(503);
    expect((await res.json()) as object).toMatchObject({ error: "origin_unavailable", samples: true });
    expect(lim.refunds).toHaveLength(1);
    const again = await handleAsk(post({ question: "q" }), env, setup(() => sseResponse(), lim).deps);
    expect(again.status).toBe(200); // the single allowed question was given back
  });

  it("origin busy (503) -> 503 busy with Retry-After, refunded", async () => {
    const lim = realLimiter();
    const { deps } = setup(() => new Response('{"error":"busy"}', { status: 503, headers: { "content-type": "application/json" } }), lim);
    const res = await handleAsk(post({ question: "q" }), env, deps);
    expect(res.status).toBe(503);
    expect(res.headers.get("retry-after")).toBe("5");
    expect((await res.json()) as object).toMatchObject({ error: "busy", samples: true });
    expect(lim.refunds).toHaveLength(1);
  });

  it("origin error or a non-stream answer -> 502 origin_error, refunded", async () => {
    for (const bad of [() => new Response("x", { status: 500 }), () => new Response("x", { status: 401 }), () => new Response("<html>", { status: 200, headers: { "content-type": "text/html" } })]) {
      const lim = realLimiter();
      const res = await handleAsk(post({ question: "q" }), env, setup(bad, lim).deps);
      expect(res.status).toBe(502);
      expect((await res.json()) as object).toMatchObject({ error: "origin_error", samples: true });
      expect(lim.refunds).toHaveLength(1);
    }
  });
});

describe("turnstile", () => {
  const withTs: Env = { ...env, TURNSTILE_SECRET_KEY: "ts-secret", TURNSTILE_SITE_KEY: "site-key" };
  const verifyBy = (ok: boolean) => (req: Request) =>
    req.url.includes("turnstile") ? new Response(JSON.stringify({ success: ok }), { headers: { "content-type": "application/json" } }) : sseResponse();

  it("is skipped when it is not configured", async () => {
    const { deps, calls } = setup();
    expect((await handleAsk(post({ question: "q" }), env, deps)).status).toBe(200);
    expect(calls.every((c) => !c.url.includes("turnstile"))).toBe(true);
    expect(((await handleConfig(env).json()) as { turnstileSiteKey: unknown }).turnstileSiteKey).toBeNull();
    // a secret without a site key (or the reverse) does not half-enable it
    expect(((await handleConfig({ ...env, TURNSTILE_SECRET_KEY: "x" }).json()) as { turnstileSiteKey: unknown }).turnstileSiteKey).toBeNull();
  });

  it("verifies the token server-side, with the secret and the client's IP, before the limiter and the origin", async () => {
    const { deps, calls } = setup(verifyBy(true));
    const res = await handleAsk(post({ question: "q", turnstile: "tok-123" }), withTs, deps);
    expect(res.status).toBe(200);
    expect(calls[0].url).toBe("https://challenges.cloudflare.com/turnstile/v0/siteverify");
    const form = new URLSearchParams(await calls[0].text());
    expect(form.get("secret")).toBe("ts-secret");
    expect(form.get("response")).toBe("tok-123");
    expect(form.get("remoteip")).toBe("203.0.113.7");
    expect(calls[1].url).toBe("https://origin.example/ask");
    expect(((await handleConfig(withTs).json()) as { turnstileSiteKey: unknown }).turnstileSiteKey).toBe("site-key");
  });

  it("403s a missing, rejected or unverifiable token, and does not spend a question", async () => {
    for (const [body, origin] of [
      [{ question: "q" }, verifyBy(true)],
      [{ question: "q", turnstile: "bad" }, verifyBy(false)],
      [{ question: "q", turnstile: "tok" }, () => { throw new Error("siteverify down"); }],
    ] as const) {
      const lim = realLimiter({ ipLimitPerHour: 1, dailyCap: 5 });
      const { deps, calls } = setup(origin, lim);
      const res = await handleAsk(post(body), withTs, deps);
      expect(res.status).toBe(403);
      expect(calls.every((c) => !c.url.startsWith("https://origin.example"))).toBe(true);
      const ok = await handleAsk(post({ question: "q", turnstile: "tok" }), withTs, setup(verifyBy(true), lim).deps);
      expect(ok.status).toBe(200); // the one allowed question was still available
    }
  });
});
