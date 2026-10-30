import { DurableObject } from "cloudflare:workers";
import { createTables, reserve, refund, type Decision, type Sql } from "./limits.ts";

interface LimiterEnv {
  IP_LIMIT_PER_HOUR?: string;
  DAILY_CAP?: string;
}

/** One instance ("global") holds both limits, in its own SQLite database. */
export class Limiter extends DurableObject<LimiterEnv> {
  private sql: Sql;

  constructor(ctx: DurableObjectState, env: LimiterEnv) {
    super(ctx, env);
    this.sql = ctx.storage.sql as unknown as Sql;
    createTables(this.sql);
  }

  private get cfg() {
    return {
      ipLimitPerHour: Number(this.env.IP_LIMIT_PER_HOUR) || 10,
      dailyCap: Number(this.env.DAILY_CAP) || 950,
    };
  }

  reserve(ip: string): Decision {
    return reserve(this.sql, this.cfg, ip);
  }

  refund(id: number, day: string): void {
    refund(this.sql, id, day);
  }
}
