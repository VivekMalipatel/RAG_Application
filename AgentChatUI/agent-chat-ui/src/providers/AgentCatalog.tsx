"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { fetchAgentCatalog, type AgentDefinition, type CapabilityToggleDefinition } from "@/lib/auth-client";
import { useAuth } from "@/providers/Auth";
import { resolveAgentIcon, resolveCapabilityIcon } from "@/lib/catalog-icons";

export type CatalogCapability = CapabilityToggleDefinition & { icon: LucideIcon };
export type CatalogAgent = AgentDefinition & {
  icon: LucideIcon;
  capabilities: CatalogCapability[];
};

type AgentCatalogContextValue = {
  catalog: CatalogAgent[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
};

const AgentCatalogContext = createContext<AgentCatalogContextValue | undefined>(undefined);

export function AgentCatalogProvider({ children }: { children: React.ReactNode }) {
  const { token } = useAuth();
  const [catalog, setCatalog] = useState<CatalogAgent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!token) {
      setCatalog([]);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAgentCatalog(token);
      const nextCatalog: CatalogAgent[] = data.map((agent) => {
        const iconOverride = agent.metadata ? agent.metadata["icon"] : undefined;
        const capabilityIconsRaw = agent.metadata ? agent.metadata["capability_icons"] : undefined;
        const capabilityIconOverrides =
          typeof capabilityIconsRaw === "object" && capabilityIconsRaw !== null
            ? (capabilityIconsRaw as Record<string, unknown>)
            : undefined;
        const capabilities = agent.capabilities.map((capability) => ({
          ...capability,
          icon: resolveCapabilityIcon(
            capability.name,
            capabilityIconOverrides?.[capability.name],
          ),
        }));
        return {
          ...agent,
          capabilities,
          icon: resolveAgentIcon(agent.agent_id, {
            icon: iconOverride,
            category: agent.category ?? null,
          }),
        };
      });
      setCatalog(nextCatalog);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load agent catalog";
      setError(message);
      setCatalog([]);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void load();
  }, [load]);

  const value = useMemo<AgentCatalogContextValue>(() => ({
    catalog,
    loading,
    error,
    refresh: load,
  }), [catalog, loading, error, load]);

  return <AgentCatalogContext.Provider value={value}>{children}</AgentCatalogContext.Provider>;
}

export function useAgentCatalog(): AgentCatalogContextValue {
  const context = useContext(AgentCatalogContext);
  if (!context) {
    throw new Error("useAgentCatalog must be used within AgentCatalogProvider");
  }
  return context;
}
