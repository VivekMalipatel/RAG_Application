"use client";

import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
  useMemo,
  useRef,
  useCallback,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message, type Thread } from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";
import {
  uiMessageReducer,
  isUIMessage,
  isRemoveUIMessage,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useThreads } from "./Thread";
import { toast } from "sonner";
import { validate } from "uuid";
import { useAuth } from "@/providers/Auth";
import { useAgentCatalog, type CatalogAgent, type CatalogCapability } from "@/providers/AgentCatalog";
import { useChatRuntime } from "@/providers/ChatRuntime";

export type StateType = {
  messages: Message[];
  ui?: UIMessage[];
  context?: Record<string, unknown>;
};

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
      context?: Record<string, unknown>;
    };
    CustomEventType: UIMessage | RemoveUIMessage;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream> & {
  capabilityFlags: Record<string, boolean>;
  setCapabilityFlag: (name: string, value: boolean) => void;
  capabilityDefinitions: CatalogCapability[];
  assistantId: string;
  assistantDefinition?: CatalogAgent;
  setAssistantId: (assistantId: string) => void;
  threadId: string | null;
  setThreadId: (threadId: string | null) => void;
};
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkGraphStatus(apiUrl: string, token: string | null): Promise<boolean> {
  try {
    const headers: Record<string, string> = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const res = await fetch(`${apiUrl}/info`, {
      headers: Object.keys(headers).length ? headers : undefined,
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiUrl,
  assistantId,
  assistantDefinition,
  orgId,
  userId,
  capabilities,
  capabilityDefinitions,
  onCapabilityChange,
  onAssistantChange,
  token,
  threadId,
  setThreadId,
}: {
  children: ReactNode;
  apiUrl: string;
  assistantId: string;
  assistantDefinition?: CatalogAgent;
  orgId: string;
  userId: string;
  capabilities: Record<string, boolean>;
  capabilityDefinitions: CatalogCapability[];
  onCapabilityChange: (name: string, value: boolean) => void;
  onAssistantChange: (assistantId: string) => void;
  token: string | null;
  threadId: string | null;
  setThreadId: (threadId: string | null) => void;
}) => {
  const { getThreads, threads, setThreads } = useThreads();
  const threadIdRef = useRef<string | null>(null);
  const resolvedCapabilities = useMemo(() => {
    if (capabilityDefinitions.length === 0) {
      return { ...capabilities };
    }
    const result: Record<string, boolean> = {};
    capabilityDefinitions.forEach((def) => {
      const value = capabilities[def.name];
      result[def.name] = typeof value === "boolean" ? value : def.enabled_by_default;
    });
    Object.entries(capabilities).forEach(([name, value]) => {
      if (!(name in result)) {
        result[name] = value;
      }
    });
    return result;
  }, [capabilities, capabilityDefinitions]);
  const configBase = useMemo(() => {
    const base: Record<string, string | boolean> = {
      org_id: orgId,
      user_id: userId,
    };
    Object.entries(resolvedCapabilities).forEach(([name, value]) => {
      base[`enable_${name}`] = value;
    });
    return base;
  }, [orgId, userId, resolvedCapabilities]);
  const metadataBase = useMemo(() => {
    const metadata: Record<string, string> = validate(assistantId)
      ? { assistant_id: assistantId }
      : { graph_id: assistantId };
    if (orgId) metadata.org_id = orgId;
    if (userId) metadata.user_id = userId;
    return metadata;
  }, [assistantId, orgId, userId]);
  useEffect(() => {
    if (!threadId) {
      threadIdRef.current = null;
    }
  }, [threadId]);
  const knownThread = useMemo(() => {
    if (!threadId) return false;
    return threads.some((item: Thread) => item.thread_id === threadId);
  }, [threads, threadId]);

  const streamThreadId = knownThread ? threadId : null;

  const streamValue = useTypedStream({
    apiUrl,
    assistantId,
    defaultHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
    threadId: streamThreadId,
    fetchStateHistory: true,
    onUpdateEvent: (data, options) => {
      options.mutate(() => {
        const updates: Partial<StateType> = {};
        let hasUpdate = false;
        Object.values(data).forEach((value) => {
          if (!value || typeof value !== "object") {
            return;
          }
          Object.entries(value as Record<string, unknown>).forEach(
            ([key, entry]) => {
              (updates as Record<string, unknown>)[key] = entry;
              hasUpdate = true;
            },
          );
        });
        return hasUpdate ? updates : {};
      });
    },
    onCustomEvent: (event, options) => {
      if (isUIMessage(event) || isRemoveUIMessage(event)) {
        options.mutate((prev) => {
          const ui = uiMessageReducer(prev.ui ?? [], event);
          return { ...prev, ui };
        });
      }
    },
    onThreadId: (id) => {
      if (id) {
        threadIdRef.current = id;
      }
      if (id && id !== threadId) {
        setThreadId(id);
      }
      // Refetch threads list when thread ID changes.
      // Wait for some seconds before fetching so we're able to get the new thread that was created.
      sleep().then(() => getThreads().then(setThreads).catch(console.error));
    },
  });
  const ensureThreadId = useCallback(() => {
    if (threadIdRef.current) {
      return threadIdRef.current;
    }
    if (threadId) {
      threadIdRef.current = threadId;
      return threadId;
    }
    const generatedId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : uuidv4();
    threadIdRef.current = generatedId;
    setThreadId(generatedId);
    return generatedId;
  }, [threadId, setThreadId]);
  const submit = useCallback(
    (
      input: Parameters<typeof streamValue.submit>[0],
      options?: Parameters<typeof streamValue.submit>[1],
    ) => {
      const resolvedThreadId = options?.threadId ?? ensureThreadId();
      const mergedConfigurable = {
        ...configBase,
        thread_id: resolvedThreadId,
        ...(options?.config?.configurable ?? {}),
      };
      const mergedMetadata = {
        ...metadataBase,
        thread_id: resolvedThreadId,
        ...(options?.metadata ?? {}),
      };
      return streamValue.submit(input, {
        ...options,
        threadId: resolvedThreadId,
        config: {
          ...(options?.config ?? {}),
          configurable: mergedConfigurable,
        },
        metadata: mergedMetadata,
      });
    },
    [streamValue, ensureThreadId, configBase, metadataBase],
  );
  const streamContextValue = useMemo(
    () => ({
      ...streamValue,
      submit,
      capabilityFlags: resolvedCapabilities,
      setCapabilityFlag: onCapabilityChange,
      capabilityDefinitions,
      assistantId,
      assistantDefinition,
      setAssistantId: onAssistantChange,
      threadId,
      setThreadId,
    }),
    [
      streamValue,
      submit,
      resolvedCapabilities,
      onCapabilityChange,
      capabilityDefinitions,
      assistantId,
      assistantDefinition,
      onAssistantChange,
      threadId,
      setThreadId,
    ],
  );

  useEffect(() => {
    checkGraphStatus(apiUrl, token).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to LangGraph server", {
          description: () => (
            <p>
              Please ensure your graph is running at <code>{apiUrl}</code> and that
              your account has access to this deployment.
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiUrl, token]);

  return (
    <StreamContext.Provider value={streamContextValue}>
      {children}
    </StreamContext.Provider>
  );
};

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const envUserId: string | undefined = process.env.NEXT_PUBLIC_USER_ID;

  const { token, user } = useAuth();
  const { catalog } = useAgentCatalog();
  const {
    apiUrl,
    orgId,
    assistantId: runtimeAssistantId,
    setAssistantId: setRuntimeAssistantId,
    threadId,
    setThreadId,
  } = useChatRuntime();

  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;

  const assistantId = runtimeAssistantId || envAssistantId || "";
  const userId = user?.id ?? envUserId ?? "";

  const assistantDefinition = useMemo(
    () => catalog.find((item) => item.agent_id === assistantId),
    [catalog, assistantId],
  );
  const capabilityDefinitions = useMemo(
    () => assistantDefinition?.capabilities ?? [],
    [assistantDefinition],
  );
  const [capabilityFlags, setCapabilityFlags] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!assistantId && catalog.length > 0) {
      setRuntimeAssistantId(catalog[0].agent_id);
    }
  }, [assistantId, catalog, setRuntimeAssistantId]);

  useEffect(() => {
    if (capabilityDefinitions.length === 0) {
      setCapabilityFlags((prev) => (Object.keys(prev).length === 0 ? prev : {}));
      return;
    }
    setCapabilityFlags((prev) => {
      const next: Record<string, boolean> = {};
      capabilityDefinitions.forEach((definition) => {
        const current = prev[definition.name];
        next[definition.name] =
          typeof current === "boolean" ? current : definition.enabled_by_default;
      });
      const unchanged =
        Object.keys(prev).length === Object.keys(next).length &&
        capabilityDefinitions.every((definition) => prev[definition.name] === next[definition.name]);
      return unchanged ? prev : next;
    });
  }, [capabilityDefinitions]);

  const handleCapabilityChange = useCallback(
    (name: string, value: boolean) => {
      setCapabilityFlags((prev) => {
        if (prev[name] === value) return prev;
        return { ...prev, [name]: value };
      });
    },
    [],
  );

  const handleAssistantChange = useCallback(
    (nextAssistantId: string) => {
      if (nextAssistantId === assistantId) {
        return;
      }
      setRuntimeAssistantId(nextAssistantId);
      setThreadId(null);
      setCapabilityFlags({});
    },
    [assistantId, setRuntimeAssistantId, setThreadId],
  );

  return (
    <StreamSession
      apiUrl={apiUrl}
      assistantId={assistantId}
      assistantDefinition={assistantDefinition}
      orgId={orgId}
      userId={userId}
      capabilities={capabilityFlags}
      capabilityDefinitions={capabilityDefinitions}
      onCapabilityChange={handleCapabilityChange}
      onAssistantChange={handleAssistantChange}
      token={token ?? null}
      threadId={threadId}
      setThreadId={setThreadId}
    >
      {children}
    </StreamSession>
  );
};

// Create a custom hook to use the context
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
