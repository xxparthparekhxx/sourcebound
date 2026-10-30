import { beforeEach, describe, expect, it } from "vitest";
import { createTables, refund, reserve, type Sql } from "../src/limits.ts";
import { memorySql } from "./sqlite.ts";

const cfg = { ipLimitPerHour: 10, dailyCap: 950 };
const T0 = Date.UTC(2026, 9, 30, 12, 0, 0);
let sql: Sql;

beforeEach(() => {
  sql = memorySql();
  createTables(sql);
});

describe("per-client hourly limit", () => {
  it("allows 10 questions an hour and refuses the 11th", () => {
    for (let i = 0; i < 10; i++) expect(reserve(sql, cfg, "a", T0 + i * 1000).ok).toBe(true);
    const d = reserve(sql, cfg, "a", T0 + 11_000);
    expect(d).toMatchObject({ ok: false, scope: "ip" });
  });

  it("does not count other clients", () => {
    for (let i = 0; i < 10; i++) reserve(sql, cfg, "a", T0);
    expect(reserve(sql, cfg, "b", T0).ok).toBe(true);
  });

  it("is a sliding window: the oldest question ages out after an hour, and retryAfter says when", () => {
    for (let i = 0; i < 10; i++) reserve(sql, cfg, "a", T0 + i * 60_000); // one a minute
    const denied = reserve(sql, cfg, "a", T0 + 10 * 60_000);
    expect(denied).toMatchObject({ ok: false, scope: "ip", retryAfter: 3000 }); // oldest at T0 -> free at T0+3600s, now T0+600s
    expect(reserve(sql, cfg, "a", T0 + 3_600_000 - 1).ok).toBe(false);
    expect(reserve(sql, cfg, "a", T0 + 3_600_000 + 1).ok).toBe(true);
    expect(reserve(sql, cfg, "a", T0 + 3_600_000 + 2).ok).toBe(false); // only one slot opened
  });

  it("a refused request is not counted", () => {
    for (let i = 0; i < 10; i++) reserve(sql, cfg, "a", T0);
    for (let i = 0; i < 5; i++) reserve(sql, cfg, "a", T0 + 1000);
    const rows = sql.exec("SELECT COUNT(*) AS n FROM hits").toArray();
    expect(rows[0].n).toBe(10);
    expect(sql.exec("SELECT n FROM daily").toArray()[0].n).toBe(10);
  });
});

describe("global daily cap", () => {
  it("stops at 950 across clients and reports the time to the UTC reset", () => {
    for (let i = 0; i < 950; i++) expect(reserve(sql, cfg, `ip${i % 200}`, T0 + i).ok, `question ${i + 1}`).toBe(true);
    const d = reserve(sql, cfg, "fresh-client", T0 + 1000);
    expect(d).toMatchObject({ ok: false, scope: "daily" });
    expect((d as { retryAfter: number }).retryAfter).toBe(12 * 3600 - 1); // 12:00:01 -> 24:00:00
  });

  it("checks the per-client limit first, so a limited client is told so", () => {
    const small = { ipLimitPerHour: 2, dailyCap: 2 };
    reserve(sql, small, "a", T0);
    reserve(sql, small, "a", T0);
    expect(reserve(sql, small, "a", T0)).toMatchObject({ scope: "ip" });
    expect(reserve(sql, small, "b", T0)).toMatchObject({ scope: "daily" });
  });

  it("starts a fresh count on the next UTC day and drops the old row", () => {
    const small = { ipLimitPerHour: 10, dailyCap: 3 };
    for (let i = 0; i < 3; i++) reserve(sql, small, `c${i}`, T0);
    expect(reserve(sql, small, "d", T0).ok).toBe(false);
    const tomorrow = Date.UTC(2026, 9, 31, 0, 0, 1);
    expect(reserve(sql, small, "d", tomorrow)).toMatchObject({ ok: true, used: 1, day: "2026-10-31" });
    expect(sql.exec("SELECT day FROM daily").toArray().map((r) => r.day)).toEqual(["2026-10-31"]);
  });
});

describe("refund", () => {
  it("gives both the client's slot and the day's count back, once", () => {
    const small = { ipLimitPerHour: 1, dailyCap: 1 };
    const d = reserve(sql, small, "a", T0);
    if (!d.ok) throw new Error("expected a reservation");
    expect(reserve(sql, small, "a", T0).ok).toBe(false);
    refund(sql, d.id, d.day);
    refund(sql, d.id, d.day); // second call changes nothing
    expect(sql.exec("SELECT n FROM daily").toArray()[0].n).toBe(0);
    expect(reserve(sql, small, "a", T0).ok).toBe(true);
    expect(reserve(sql, small, "b", T0)).toMatchObject({ scope: "daily" }); // still capped at 1: no double refund
  });
});
