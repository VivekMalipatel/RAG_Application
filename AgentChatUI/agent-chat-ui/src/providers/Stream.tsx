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
import { type Message } from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";
import {
  uiMessageReducer,
  isUIMessage,
  isRemoveUIMessage,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState, parseAsBoolean } from "nuqs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { LangGraphLogoSVG } from "@/components/icons/langgraph";
import { Label } from "@/components/ui/label";
import { ArrowRight } from "lucide-react";
import { PasswordInput } from "@/components/ui/password-input";
import { getApiKey } from "@/lib/api-key";
import { useThreads } from "./Thread";
import { toast } from "sonner";
import { validate } from "uuid";

export type StateType = { messages: Message[]; ui?: UIMessage[] };

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

type StreamContextType = ReturnType<typeof useTypedStream>;
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
): Promise<boolean> {
  try {
    const res = await fetch(`${apiUrl}/info`, {
      ...(apiKey && {
        headers: {
          "X-Api-Key": apiKey,
        },
      }),
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl,
  assistantId,
  orgId,
  userId,
  enableKnowledgeSearch,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string;
  assistantId: string;
  orgId: string;
  userId: string;
  enableKnowledgeSearch: boolean;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, setThreads } = useThreads();
  const threadIdRef = useRef<string | null>(null);
  const configBase = useMemo(
    () => ({
      org_id: orgId,
      user_id: userId,
      enable_knowledge_search: enableKnowledgeSearch,
    }),
    [orgId, userId, enableKnowledgeSearch],
  );
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
  const streamValue = useTypedStream({
    apiUrl,
    apiKey: apiKey ?? undefined,
    assistantId,
    threadId: threadId ?? null,
    fetchStateHistory: true,
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
    }),
    [streamValue, submit],
  );

  useEffect(() => {
    checkGraphStatus(apiUrl, apiKey).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to LangGraph server", {
          description: () => (
            <p>
              Please ensure your graph is running at <code>{apiUrl}</code> and
              your API key is correctly set (if connecting to a deployed graph).
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiKey, apiUrl]);

  return (
    <StreamContext.Provider value={streamContextValue}>
      {children}
    </StreamContext.Provider>
  );
};

// Default values for the form
const DEFAULT_API_URL = "http://localhost:8123";
const DEFAULT_ASSISTANT_ID = "chat_agent";
const DEFAULT_ORG_ID = "local-org";
const DEFAULT_USER_ID = "local-user";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  // Get environment variables
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envOrgId: string | undefined = process.env.NEXT_PUBLIC_ORG_ID;
  const envUserId: string | undefined = process.env.NEXT_PUBLIC_USER_ID;
  const envEnableKnowledgeSearch =
    process.env.NEXT_PUBLIC_ENABLE_KNOWLEDGE_SEARCH === "true";

  // Use URL params with env var fallbacks
  const [apiUrl, setApiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId, setAssistantId] = useQueryState("assistantId", {
    defaultValue: envAssistantId || "",
  });
  const [orgId, setOrgId] = useQueryState("orgId", {
    defaultValue: envOrgId || "",
  });
  const [userId, setUserId] = useQueryState("userId", {
    defaultValue: envUserId || "",
  });
  const [enableKnowledgeSearch, setEnableKnowledgeSearch] = useQueryState(
    "enableKnowledgeSearch",
    parseAsBoolean.withDefault(envEnableKnowledgeSearch),
  );

  // For API key, use localStorage with env var fallback
  const [apiKey, _setApiKey] = useState(() => {
    const storedKey = getApiKey();
    return storedKey || "";
  });

  const setApiKey = (key: string) => {
    window.localStorage.setItem("lg:chat:apiKey", key);
    _setApiKey(key);
  };

  // Determine final values to use, prioritizing URL params then env vars
  const finalApiUrl = apiUrl || envApiUrl;
  const finalAssistantId = assistantId || envAssistantId;
  const finalOrgId = orgId || envOrgId;
  const finalUserId = userId || envUserId;
  const finalEnableKnowledgeSearch = enableKnowledgeSearch;

  // Show the form if we: don't have an API URL, or don't have an assistant ID
  if (!finalApiUrl || !finalAssistantId || !finalOrgId || !finalUserId) {
    return (
      <div className="flex min-h-screen w-full items-center justify-center p-4">
        <div className="animate-in fade-in-0 zoom-in-95 bg-background flex max-w-3xl flex-col rounded-lg border shadow-lg">
          <div className="mt-14 flex flex-col gap-2 border-b p-6">
            <div className="flex flex-col items-start gap-2">
              <LangGraphLogoSVG className="h-7" />
              <h1 className="text-xl font-semibold tracking-tight">
                Agent Chat
              </h1>
            </div>
            <p className="text-muted-foreground">
              Welcome to Agent Chat! Before you get started, you need to enter
              the URL of the deployment and the assistant / graph ID.
            </p>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();

              const form = e.target as HTMLFormElement;
              const formData = new FormData(form);
              const apiUrl = formData.get("apiUrl") as string;
              const assistantId = formData.get("assistantId") as string;
              const apiKey = formData.get("apiKey") as string;
              const orgId = formData.get("orgId") as string;
              const userId = formData.get("userId") as string;
              setApiUrl(apiUrl);
              setApiKey(apiKey);
              setAssistantId(assistantId);
              setOrgId(orgId);
              setUserId(userId);

              form.reset();
            }}
            className="bg-muted/50 flex flex-col gap-6 p-6"
          >
            <div className="flex flex-col gap-2">
              <Label htmlFor="apiUrl">
                Deployment URL<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                This is the URL of your LangGraph deployment. Can be a local, or
                production deployment.
              </p>
              <Input
                id="apiUrl"
                name="apiUrl"
                className="bg-background"
                defaultValue={apiUrl || envApiUrl || DEFAULT_API_URL}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="assistantId">
                Assistant / Graph ID<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                This is the ID of the graph (can be the graph name), or
                assistant to fetch threads from, and invoke when actions are
                taken.
              </p>
              <Input
                id="assistantId"
                name="assistantId"
                className="bg-background"
                defaultValue={
                  assistantId || envAssistantId || DEFAULT_ASSISTANT_ID
                }
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="orgId">
                Organization ID<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                Threads and knowledge lookups are scoped to this organization.
              </p>
              <Input
                id="orgId"
                name="orgId"
                className="bg-background"
                defaultValue={orgId || envOrgId || DEFAULT_ORG_ID}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="userId">
                User ID<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                The chat session will be attributed to this user.
              </p>
              <Input
                id="userId"
                name="userId"
                className="bg-background"
                defaultValue={userId || envUserId || DEFAULT_USER_ID}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="apiKey">LangSmith API Key</Label>
              <p className="text-muted-foreground text-sm">
                This is <strong>NOT</strong> required if using a local LangGraph
                server. This value is stored in your browser's local storage and
                is only used to authenticate requests sent to your LangGraph
                server.
              </p>
              <PasswordInput
                id="apiKey"
                name="apiKey"
                defaultValue={apiKey ?? ""}
                className="bg-background"
                placeholder="lsv2_pt_..."
              />
            </div>
            <div className="mt-2 flex justify-end">
              <Button
                type="submit"
                size="lg"
              >
                Continue
                <ArrowRight className="size-5" />
              </Button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  return (
    <StreamSession
      apiKey={apiKey}
      apiUrl={finalApiUrl!}
      assistantId={finalAssistantId!}
      orgId={finalOrgId!}
      userId={finalUserId!}
      enableKnowledgeSearch={finalEnableKnowledgeSearch}
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
