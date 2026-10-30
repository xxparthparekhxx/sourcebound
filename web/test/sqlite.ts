import { DatabaseSync } from "node:sqlite";
import type { Sql } from "../src/limits.ts";

/** The Durable Object's `sql.exec` interface on top of node:sqlite, so the limit logic is tested as written. */
export function memorySql(): Sql {
  const db = new DatabaseSync(":memory:");
  return {
    exec(query: string, ...bindings: unknown[]) {
      const rows = db.prepare(query).all(...(bindings as never[])) as Record<string, unknown>[];
      return { toArray: () => rows };
    },
  };
}
