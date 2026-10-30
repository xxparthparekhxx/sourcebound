import type { Decision } from "./limits.ts";

export const MAX_QUESTION_CHARS = 500;
const MAX_BODY_BYTES = 4096;
const ORIGIN_HEADER_TIMEOUT_MS = 90_000;

export interface Env {
  ORIGIN_URL: string;
  GATEWAY_SECRET?: string;
  TURNSTILE_SECRET_KEY?: string;
  TURNSTILE_SITE_KEY?: string;
  IP_LIMIT_PER_HOUR?: string;
  DAILY_CAP?: string;
}

/** What the handler needs from the limiter; the Durable Object stub implements it. */
export interface LimiterApi {
  reserve(ip: string): Promise<Decision>;
  refund(id: number, day: string): Promise<void>;
}

export interface Deps {
  limiter: LimiterApi;
  fetch: typeof fetch;
}

const json = (body: unknown, status: number, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...headers },
  });

/** Every failure the page can show a saved answer for carries `samples: true`. */
const fail = (error: string, message: string, status: number, extra: Record<string, unknown> = {}, headers: Record<string, string> = {}) =>
  json({ error, message, samples: true, ...extra }, status, headers);

export function turnstileEnabled(env: Env): boolean {
  return Boolean(env.TURNSTILE_SECRET_KEY && env.TURNSTILE_SITE_KEY);
}

async function readJson(request: Request): Promise<unknown | Response> {
  const declared = Number(request.headers.get("content-length") ?? 0);
  if (declared > MAX_BODY_BYTES) return json({ error: "too_large", message: "The request is too large." }, 413);
  const text = await request.text();
  if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) return json({ error: "too_large", message: "The request is too large." }, 413);
  try {
    return JSON.parse(text);
  } catch {
    return json({ error: "bad_request", message: "The request body must be JSON." }, 400);
  }
}

async function verifyTurnstile(env: Env, token: unknown, ip: string, doFetch: typeof fetch): Promise<boolean> {
  if (typeof token !== "string" || token.length === 0 || token.length > 2048) return false;
  const form = new URLSearchParams({ secret: env.TURNSTILE_SECRET_KEY ?? "", response: token });
  if (ip) form.set("remoteip", ip);
  try {
    const r = await doFetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", { method: "POST", body: form });
    const body = (await r.json()) as { success?: boolean };
    return body.success === true;
  } catch {
    return false;
  }
}

async function hashIp(ip: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`sourcebound:${ip}`));
  return [...new Uint8Array(digest)].slice(0, 12).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function handleAsk(request: Request, env: Env, deps: Deps): Promise<Response> {
  if (request.method !== "POST") return json({ error: "method_not_allowed", message: "Use POST." }, 405, { allow: "POST" });
  if (!env.GATEWAY_SECRET || !env.ORIGIN_URL) return fail("not_configured", "The demo is not configured.", 503);

  const body = await readJson(request);
  if (body instanceof Response) return body;
  if (typeof body !== "object" || body === null || Array.isArray(body)) return json({ error: "bad_request", message: "Send {\"question\": \"...\"}." }, 400);
  const { question: raw, turnstile } = body as { question?: unknown; turnstile?: unknown };
  if (typeof raw !== "string" || raw.trim().length === 0) return json({ error: "bad_request", message: "Send a question." }, 400);
  const question = raw.trim();
  if (question.length > MAX_QUESTION_CHARS) {
    return json({ error: "too_long", message: `Questions are limited to ${MAX_QUESTION_CHARS} characters.` }, 413);
  }

  const ip = request.headers.get("cf-connecting-ip") ?? "";
  if (turnstileEnabled(env) && !(await verifyTurnstile(env, turnstile, ip, deps.fetch))) {
    return json({ error: "turnstile", message: "The bot check failed. Reload the page and try again." }, 403);
  }

  const decision = await deps.limiter.reserve(await hashIp(ip || "unknown"));
  if (!decision.ok) {
    const wait = decision.retryAfter >= 3600 ? `${Math.ceil(decision.retryAfter / 3600)} hours` : `${Math.max(1, Math.ceil(decision.retryAfter / 60))} minutes`;
    return decision.scope === "ip"
      ? fail("limit_ip", `You have used your ${env.IP_LIMIT_PER_HOUR ?? 10} questions for this hour. Try again in about ${wait}.`, 429, { scope: "ip", retryAfter: decision.retryAfter }, { "retry-after": String(decision.retryAfter) })
      : fail("limit_daily", `The demo has used its free language-model quota for today. It resets at 00:00 UTC (in about ${wait}).`, 429, { scope: "daily", retryAfter: decision.retryAfter }, { "retry-after": String(decision.retryAfter) });
  }
  const giveBack = () => deps.limiter.refund(decision.id, decision.day).catch(() => undefined);

  let upstream: Response;
  try {
    upstream = await deps.fetch(new URL("/ask", env.ORIGIN_URL).toString(), {
      method: "POST",
      headers: { "content-type": "application/json", "x-gateway-secret": env.GATEWAY_SECRET, accept: "text/event-stream" },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(ORIGIN_HEADER_TIMEOUT_MS),
    });
  } catch (err) {
    console.warn("origin fetch failed:", err instanceof Error ? `${err.name}: ${err.message}` : "unknown");
    await giveBack();
    return fail("origin_unavailable", "The server that runs the models is not reachable right now.", 503);
  }

  if (upstream.status === 503) {
    await upstream.body?.cancel();
    await giveBack();
    return fail("busy", "The demo is busy answering other questions. Try again in a few seconds.", 503, {}, { "retry-after": "5" });
  }
  const type = upstream.headers.get("content-type") ?? "";
  if (upstream.status !== 200 || !type.startsWith("text/event-stream") || !upstream.body) {
    await upstream.body?.cancel();
    await giveBack();
    return fail("origin_error", "The server that runs the models returned an error.", 502);
  }
  return new Response(upstream.body, {
    status: 200,
    headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache, no-transform", "x-accel-buffering": "no" },
  });
}

export function handleConfig(env: Env): Response {
  return json(
    {
      turnstileSiteKey: turnstileEnabled(env) ? env.TURNSTILE_SITE_KEY : null,
      maxQuestionChars: MAX_QUESTION_CHARS,
      ipLimitPerHour: Number(env.IP_LIMIT_PER_HOUR) || 10,
    },
    200,
    { "cache-control": "public, max-age=60" },
  );
}
