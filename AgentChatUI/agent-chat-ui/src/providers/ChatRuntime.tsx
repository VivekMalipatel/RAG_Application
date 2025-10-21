"use client";

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type ChatRuntimeContextValue = {
  apiUrl: string;
  orgId: string;
  assistantId: string;
  setAssistantId: (assistantId: string) => void;
  threadId: string | null;
  setThreadId: (threadId: string | null) => void;
};

const ChatRuntimeContext = createContext<ChatRuntimeContextValue | undefined>(
  undefined,
);

const DEFAULT_API_URL = "http://localhost:8123";
const DEFAULT_ORG_ID = "local-org";

export function ChatRuntimeProvider({
  children,
}: {
  children: ReactNode;
}) {
  const envApiUrl = process.env.NEXT_PUBLIC_API_URL || "";
  const envOrgId = process.env.NEXT_PUBLIC_ORG_ID || "";
  const envAssistantId = process.env.NEXT_PUBLIC_ASSISTANT_ID || "";

  const [assistantId, setAssistantId] = useState(envAssistantId);
  const [threadId, setThreadId] = useState<string | null>(null);

  const value = useMemo<ChatRuntimeContextValue>(
    () => ({
      apiUrl: envApiUrl || DEFAULT_API_URL,
      orgId: envOrgId || DEFAULT_ORG_ID,
      assistantId,
      setAssistantId,
      threadId,
      setThreadId,
    }),
    [assistantId, envApiUrl, envOrgId, threadId],
  );

  return (
    <ChatRuntimeContext.Provider value={value}>
      {children}
    </ChatRuntimeContext.Provider>
  );
}

export function useChatRuntime(): ChatRuntimeContextValue {
  const context = useContext(ChatRuntimeContext);
  if (!context) {
    throw new Error("useChatRuntime must be used within a ChatRuntimeProvider");
  }
  return context;
}
