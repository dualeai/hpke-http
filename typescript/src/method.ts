const METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"] as const;
const METHOD_SET: ReadonlySet<string> = new Set(METHODS);

/** HTTP methods accepted by protocol version 3. */
export type Method = (typeof METHODS)[number];

export function isMethod(value: string): value is Method {
  return METHOD_SET.has(value);
}
