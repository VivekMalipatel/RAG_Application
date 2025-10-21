"use client";

import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  LoginResult,
  UserRecord,
  createUserRequest,
  fetchCurrentUser,
  listUsersRequest,
  loginRequest,
  logoutRequest,
  updateUserRolesRequest,
} from "@/lib/auth-client";

type StoredSession = {
  token: string;
  expiresAt: string;
};

type AuthContextValue = {
  user: UserRecord | null;
  token: string | null;
  expiresAt: string | null;
  loading: boolean;
  pending: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
  isAdmin: boolean;
};

const STORAGE_KEY = "agent.auth.session";

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function readStoredSession(): StoredSession | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredSession;
  } catch {
    return null;
  }
}

function writeStoredSession(session: StoredSession | null) {
  if (typeof window === "undefined") return;
  if (!session) {
    window.localStorage.removeItem(STORAGE_KEY);
  } else {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }
}

function isExpired(expiresAt: string | null): boolean {
  if (!expiresAt) return true;
  return new Date(expiresAt).getTime() <= Date.now();
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserRecord | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [expiresAt, setExpiresAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hydratingRef = useRef(false);

  const reset = useCallback(() => {
    setUser(null);
    setToken(null);
    setExpiresAt(null);
    setError(null);
    writeStoredSession(null);
  }, []);

  const hydrate = useCallback(async () => {
    const stored = readStoredSession();
    if (!stored) {
      setLoading(false);
      return;
    }
    if (isExpired(stored.expiresAt)) {
      reset();
      setLoading(false);
      return;
    }
    hydratingRef.current = true;
    try {
      const current = await fetchCurrentUser(stored.token);
      setToken(stored.token);
      setExpiresAt(stored.expiresAt);
      setUser(current);
    } catch {
      reset();
    } finally {
      hydratingRef.current = false;
      setLoading(false);
    }
  }, [reset]);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  const login = useCallback(async (email: string, password: string) => {
    setPending(true);
    setError(null);
    try {
      const result: LoginResult = await loginRequest(email, password);
      setUser(result.user);
      setToken(result.token);
      setExpiresAt(result.expires_at);
      writeStoredSession({ token: result.token, expiresAt: result.expires_at });
    } catch (err) {
      reset();
      const message = err instanceof Error ? err.message : "Login failed";
      setError(message);
      throw err;
    } finally {
      setPending(false);
    }
  }, [reset]);

  const logout = useCallback(async () => {
    const currentToken = token;
    reset();
    if (!currentToken) return;
    try {
      await logoutRequest(currentToken);
    } catch {
      return;
    }
  }, [token, reset]);

  useEffect(() => {
    if (!expiresAt) return;
    if (isExpired(expiresAt)) {
      void logout();
    }
  }, [expiresAt, logout]);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      const current = await fetchCurrentUser(token);
      setUser(current);
    } catch {
      await logout();
    }
  }, [token, logout]);

  const value = useMemo<AuthContextValue>(() => ({
    user,
    token,
    expiresAt,
    loading,
    pending,
    error,
    login,
    logout,
    refresh,
    isAdmin: !!user?.roles.includes("admin"),
  }), [user, token, expiresAt, loading, pending, error, login, logout, refresh]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}

export async function adminListUsers(token: string) {
  return listUsersRequest(token);
}

export async function adminCreateUser(
  token: string,
  payload: { email: string; password: string; full_name?: string | null; roles?: string[] },
) {
  return createUserRequest(token, payload);
}

export async function adminUpdateUserRoles(token: string, userId: string, roles: string[]) {
  return updateUserRolesRequest(token, userId, roles);
}