// Request limits: a sliding one-hour window per client and a global cap per UTC day.
// The logic works on a minimal SQL interface so the Durable Object (SQLite storage) and the unit tests
// (node:sqlite) run exactly the same code.

export interface Sql {
  exec(query: string, ...bindings: unknown[]): { toArray(): Record<string, unknown>[] };
}

export type Decision =
  | { ok: true; id: number; day: string; used: number }
  | { ok: false; scope: "ip" | "daily"; retryAfter: number };

export interface LimitConfig {
  ipLimitPerHour: number;
  dailyCap: number;
}

const HOUR = 3_600_000;
const DAY = 86_400_000;

export function createTables(sql: Sql): void {
  sql.exec("CREATE TABLE IF NOT EXISTS hits (id INTEGER PRIMARY KEY AUTOINCREMENT, ip TEXT NOT NULL, ts INTEGER NOT NULL)");
  sql.exec("CREATE INDEX IF NOT EXISTS hits_ip_ts ON hits (ip, ts)");
  sql.exec("CREATE TABLE IF NOT EXISTS daily (day TEXT PRIMARY KEY, n INTEGER NOT NULL)");
}

const dayOf = (now: number) => new Date(now).toISOString().slice(0, 10);
const num = (row: Record<string, unknown> | undefined, key: string) => Number(row?.[key] ?? 0);

/** Check both limits and, when they pass, count the request. Runs inside one Durable Object, so it is serialised. */
export function reserve(sql: Sql, cfg: LimitConfig, ip: string, now = Date.now()): Decision {
  const day = dayOf(now);
  sql.exec("DELETE FROM hits WHERE ts <= ?", now - HOUR);
  sql.exec("DELETE FROM daily WHERE day < ?", day);

  const mine = sql.exec("SELECT COUNT(*) AS n, MIN(ts) AS oldest FROM hits WHERE ip = ?", ip).toArray()[0];
  if (num(mine, "n") >= cfg.ipLimitPerHour) {
    const retryAfter = Math.max(1, Math.ceil((num(mine, "oldest") + HOUR - now) / 1000));
    return { ok: false, scope: "ip", retryAfter };
  }
  const used = num(sql.exec("SELECT n FROM daily WHERE day = ?", day).toArray()[0], "n");
  if (used >= cfg.dailyCap) {
    const nextUtcMidnight = Math.floor(now / DAY) * DAY + DAY;
    return { ok: false, scope: "daily", retryAfter: Math.max(1, Math.ceil((nextUtcMidnight - now) / 1000)) };
  }
  sql.exec("INSERT INTO hits (ip, ts) VALUES (?, ?)", ip, now);
  const id = num(sql.exec("SELECT last_insert_rowid() AS id").toArray()[0], "id");
  sql.exec("INSERT INTO daily (day, n) VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET n = n + 1", day);
  return { ok: true, id, day, used: used + 1 };
}

/** Give a reservation back (the origin never reached the language model). */
export function refund(sql: Sql, id: number, day: string): void {
  const gone = sql.exec("SELECT COUNT(*) AS n FROM hits WHERE id = ?", id).toArray()[0];
  if (num(gone, "n") === 0) return; // already refunded, or aged out of the window
  sql.exec("DELETE FROM hits WHERE id = ?", id);
  sql.exec("UPDATE daily SET n = MAX(n - 1, 0) WHERE day = ?", day);
}
