function resolveBaseUrl(): string {
  const envBase = process.env.NEXT_PUBLIC_AGENT_API_BASE_URL;
  if (envBase) return envBase.replace(/\/$/, "");
  if (typeof window !== "undefined" && window.location) {
    const { protocol, hostname } = window.location;
    const port = process.env.NEXT_PUBLIC_AGENT_API_PORT ?? "8000";
    const finalPort = port && port !== "80" && port !== "443" ? `:${port}` : "";
    return `${protocol}//${hostname}${finalPort}`.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

const BASE_URL = resolveBaseUrl();

type RequestConfig = {
  method?: string;
  token?: string;
  body?: unknown;
};

type ApiError = {
  detail?: string;
  message?: string;
};

async function request<T>(path: string, config: RequestConfig = {}): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (config.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (config.token) {
    headers.Authorization = `Bearer ${config.token}`;
  }
  const response = await fetch(`${BASE_URL}${path}`, {
    method: config.method ?? "GET",
    headers,
    body: config.body !== undefined ? JSON.stringify(config.body) : undefined,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const data = (await response.json()) as ApiError;
      if (data.detail) message = Array.isArray(data.detail) ? data.detail.join(", ") : `${data.detail}`;
      else if (data.message) message = data.message;
    } catch {
      message = response.statusText;
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export type UserRecord = {
  id: string;
  email: string;
  full_name: string | null;
  roles: string[];
  is_active: boolean;
  created_at: string;
};

export type LoginResult = {
  token: string;
  expires_at: string;
  user: UserRecord;
};

export type CapabilityToggleDefinition = {
  name: string;
  label: string;
  description?: string | null;
  enabled_by_default: boolean;
};

export type AgentDefinition = {
  agent_id: string;
  display_name: string;
  description: string | null;
  category: string | null;
  capabilities: CapabilityToggleDefinition[];
  metadata: Record<string, unknown>;
};

export async function loginRequest(email: string, password: string): Promise<LoginResult> {
  return request<LoginResult>("/auth/login", {
    method: "POST",
    body: { email, password },
  });
}

export async function logoutRequest(token: string): Promise<void> {
  await request<void>("/auth/logout", {
    method: "POST",
    token,
  });
}

export async function fetchCurrentUser(token: string): Promise<UserRecord> {
  return request<UserRecord>("/auth/me", {
    token,
  });
}

export async function fetchAgentCatalog(token: string): Promise<AgentDefinition[]> {
  return request<AgentDefinition[]>("/agents/catalog", {
    token,
  });
}

export async function listUsersRequest(token: string): Promise<UserRecord[]> {
  return request<UserRecord[]>("/admin/users", {
    token,
  });
}

export async function createUserRequest(
  token: string,
  payload: { email: string; password: string; full_name?: string | null; roles?: string[] },
): Promise<UserRecord> {
  return request<UserRecord>("/admin/users", {
    method: "POST",
    token,
    body: payload,
  });
}

export async function updateUserRolesRequest(
  token: string,
  userId: string,
  roles: string[],
): Promise<UserRecord> {
  return request<UserRecord>(`/admin/users/${userId}/roles`, {
    method: "PUT",
    token,
    body: { roles },
  });
}