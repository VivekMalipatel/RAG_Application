import { Client } from "@langchain/langgraph-sdk";

export function createClient(
  apiUrl: string,
  apiKey: string | undefined,
  token: string | null,
) {
  return new Client({
    apiKey,
    apiUrl,
    defaultHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
  });
}
