import pino from "pino";

import { config } from "./config.js";

const pretty = config.ENV === "development" && process.env.VITEST !== "true";

export const logger = pino({
  level: config.LOG_LEVEL,
  ...(pretty
    ? { transport: { target: "pino-pretty", options: { colorize: true, translateTime: "SYS:standard" } } }
    : {}),
});

/**
 * Single greppable event for any outbound paid/provider call. Nothing calls this
 * in Stage 0 - it exists so every billable call from here on is auditable with:
 *   docker compose logs api | grep external_call
 */
export function logExternalCall(
  provider: string,
  operation: string,
  extra: Record<string, unknown> = {},
): void {
  logger.info({ event: "external_call", provider, operation, ...extra });
}
