import { handleAsk, handleConfig, type Env as AskEnv, type LimiterApi } from "./ask.ts";
import type { Limiter } from "./limiter.ts";

export { Limiter } from "./limiter.ts";

interface Env extends AskEnv {
  ASSETS: Fetcher;
  LIMITER: DurableObjectNamespace<Limiter>;
}

function limiterFor(env: Env): LimiterApi {
  const stub = env.LIMITER.get(env.LIMITER.idFromName("global"));
  return { reserve: (ip) => stub.reserve(ip), refund: (id, day) => stub.refund(id, day) };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);
    if (pathname === "/api/ask") return handleAsk(request, env, { limiter: limiterFor(env), fetch: (input, init) => fetch(input, init) });
    if (pathname === "/api/config") return handleConfig(env);
    if (pathname.startsWith("/api/")) return new Response(JSON.stringify({ error: "not_found" }), { status: 404, headers: { "content-type": "application/json" } });
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
