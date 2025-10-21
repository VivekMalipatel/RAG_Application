import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import type { StreamAuditEvent } from "@/providers/Stream";
import { useAgentCatalog } from "@/providers/AgentCatalog";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
  Plus,
  Bug,
  Globe2,
  ChevronDown,
  Check,
  Sparkles,
  Settings,
  LogOut,
  Trash2,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { GitHubSVG } from "../icons/github";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { useFileUpload } from "@/hooks/use-file-upload";
import { ContentBlocksPreview } from "./ContentBlocksPreview";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";
import { useAuth } from "@/providers/Auth";
import { useRouter } from "next/navigation";
import { useChatRuntime } from "@/providers/ChatRuntime";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

function OpenGitHubRepo() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href="https://github.com/langchain-ai/agent-chat-ui"
            target="_blank"
            className="flex items-center justify-center"
          >
            <GitHubSVG
              width="24"
              height="24"
            />
          </a>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Open GitHub repo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const formatAuditPayload = (value: unknown): string => {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const { threadId, setThreadId: updateThreadId } = useChatRuntime();
  const [chatHistoryOpen, setChatHistoryOpen] = useState(false);
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [auditPanelOpen, setAuditPanelOpen] = useQueryState(
    "streamInspector",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks: _resetBlocks,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const { catalog } = useAgentCatalog();

  const stream = useStreamContext();
  const {
    clearAuditEvents,
    auditEvents,
    capabilityDefinitions,
    capabilityFlags,
    setCapabilityFlag,
    assistantId,
    assistantDefinition,
    setAssistantId,
  } = stream;
  const [capabilityMenuOpen, setCapabilityMenuOpen] = useState(false);
  const capabilityMenuRef = useRef<HTMLDivElement | null>(null);
  const capabilityButtonRef = useRef<HTMLButtonElement | null>(null);
  const [agentMenuOpen, setAgentMenuOpen] = useState(false);
  const agentMenuRef = useRef<HTMLDivElement | null>(null);
  const agentButtonRef = useRef<HTMLButtonElement | null>(null);
  const selectedAgent = assistantDefinition ?? catalog.find((item) => item.agent_id === assistantId);
  const AgentIcon = selectedAgent?.icon ?? Sparkles;

  const messages = stream.messages;
  const isLoading = stream.isLoading;

  const lastError = useRef<string | undefined>(undefined);

  const setThreadId = (id: string | null) => {
    updateThreadId(id);

    // close artifact and reset artifact context
    closeArtifact();
    setArtifactContext({});
  };

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      const target = event.target as Node;
      if (
        capabilityMenuOpen &&
        capabilityMenuRef.current &&
        !capabilityMenuRef.current.contains(target) &&
        capabilityButtonRef.current &&
        !capabilityButtonRef.current.contains(target)
      ) {
        setCapabilityMenuOpen(false);
      }
      if (
        agentMenuOpen &&
        agentMenuRef.current &&
        !agentMenuRef.current.contains(target) &&
        agentButtonRef.current &&
        !agentButtonRef.current.contains(target)
      ) {
        setAgentMenuOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        if (capabilityMenuOpen) {
          setCapabilityMenuOpen(false);
        }
        if (agentMenuOpen) {
          setAgentMenuOpen(false);
        }
      }
    }
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [capabilityMenuOpen, agentMenuOpen]);

  useEffect(() => {
    if (!capabilityMenuOpen) {
      return;
    }
    if (capabilityDefinitions.length <= 1) {
      setCapabilityMenuOpen(false);
    }
  }, [capabilityDefinitions, capabilityMenuOpen]);

  useEffect(() => {
    setCapabilityMenuOpen(false);
    setAgentMenuOpen(false);
  }, [assistantId]);

  useEffect(() => {
    if (!agentMenuOpen) {
      return;
    }
    if (catalog.length <= 1) {
      setAgentMenuOpen(false);
    }
  }, [catalog, agentMenuOpen]);

  const capabilityMenuEnabled = capabilityDefinitions.length > 1;
  const singleCapability = capabilityDefinitions.length === 1 ? capabilityDefinitions[0] : undefined;
  const activeCapabilityCount = capabilityDefinitions.reduce((count, definition) => {
    const value = capabilityFlags[definition.name];
    if (typeof value === "boolean") {
      return value ? count + 1 : count;
    }
    return definition.enabled_by_default ? count + 1 : count;
  }, 0);
  const SingleCapabilityIcon = singleCapability?.icon ?? Globe2;
  const agentSelector = (
    <div className="relative">
      <button
        type="button"
        ref={agentButtonRef}
        onClick={() => {
          if (catalog.length > 1) {
            setCapabilityMenuOpen(false);
            setAgentMenuOpen((open) => !open);
          }
        }}
        className={cn(
          "flex items-center gap-3 rounded-full border border-transparent bg-muted px-4 py-2 text-left transition",
          catalog.length > 1 ? "cursor-pointer hover:border-border" : "cursor-default",
        )}
      >
        <span className="flex size-9 items-center justify-center rounded-full bg-background text-primary">
          <AgentIcon className="h-4 w-4" />
        </span>
        <span className="flex flex-col items-start">
          <span className="text-xs font-medium text-muted-foreground">Agent</span>
          <span className="text-sm font-semibold leading-tight">
            {selectedAgent?.display_name ?? "Select agent"}
          </span>
        </span>
        {catalog.length > 1 && (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>
      {catalog.length > 1 && agentMenuOpen && (
        <div
          ref={agentMenuRef}
          className="absolute left-0 top-full z-50 mt-2 w-72 rounded-2xl border border-border bg-background p-2 shadow-xl"
        >
          <div className="flex flex-col gap-1">
            {catalog.map((agent) => {
              const Icon = agent.icon ?? Sparkles;
              const active = agent.agent_id === assistantId;
              return (
                <button
                  key={agent.agent_id}
                  type="button"
                  onClick={() => {
                    setAgentMenuOpen(false);
                    if (agent.agent_id !== assistantId) {
                      setAssistantId(agent.agent_id);
                      setThreadId(null);
                    }
                  }}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left transition",
                    active ? "bg-muted" : "hover:bg-muted/70",
                  )}
                >
                  <span className="flex size-9 items-center justify-center rounded-full bg-background text-primary">
                    <Icon className="h-4 w-4" />
                  </span>
                  <span className="flex-1">
                    <span className="block text-sm font-semibold text-foreground">
                      {agent.display_name}
                    </span>
                    {agent.description && (
                      <span className="mt-0.5 block text-xs text-muted-foreground">
                        {agent.description}
                      </span>
                    )}
                  </span>
                  {active && <Check className="h-4 w-4 text-primary" />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading)
      return;
    setFirstTokenReceived(false);

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(input.trim().length > 0 ? [{ type: "text", text: input }] : []),
        ...contentBlocks,
      ] as Message["content"],
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);

    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    stream.submit(
      { messages: [...toolMessages, newHumanMessage], context },
      {
        streamMode: ["values"],
        streamSubgraphs: true,
        streamResumable: true,
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
    setContentBlocks([]);
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="relative flex h-screen w-full overflow-hidden">
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-white"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div
            className="relative h-full"
            style={{ width: 300 }}
          >
            <ThreadHistory
              open={chatHistoryOpen}
              onOpenChange={setChatHistoryOpen}
            />
          </div>
        </motion.div>
      </div>

      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-2 pl-4 pr-4">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                    ) : (
                      <PanelRightClose className="size-5" />
                    )}
                  </Button>
                )}
              </div>
              {agentSelector}
              <div className="flex items-center">
                <OpenGitHubRepo />
              </div>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2">
              <div className="relative flex items-center justify-start gap-3">
                <div className="absolute left-0 z-10">
                  {(!chatHistoryOpen || !isLargeScreen) && (
                    <Button
                      className="hover:bg-gray-100"
                      variant="ghost"
                      onClick={() => setChatHistoryOpen((p) => !p)}
                    >
                      {chatHistoryOpen ? (
                        <PanelRightOpen className="size-5" />
                      ) : (
                        <PanelRightClose className="size-5" />
                      )}
                    </Button>
                  )}
                </div>
                <motion.div
                  className="flex items-center gap-3"
                  animate={{
                    marginLeft: !chatHistoryOpen ? 48 : 0,
                  }}
                  transition={{
                    type: "spring",
                    stiffness: 300,
                    damping: 30,
                  }}
                >
                  <button
                    type="button"
                    className="flex items-center justify-center rounded-full bg-muted p-2 transition hover:bg-muted/80"
                    onClick={() => setThreadId(null)}
                  >
                    <LangGraphLogoSVG
                      width={24}
                      height={24}
                    />
                  </button>
                  {agentSelector}
                </motion.div>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center">
                  <OpenGitHubRepo />
                </div>
                <TooltipIconButton
                  size="lg"
                  className={cn("p-4", auditPanelOpen && "bg-muted")}
                  tooltip="Toggle stream inspector"
                  variant="ghost"
                  onClick={() => void setAuditPanelOpen((value) => !value)}
                >
                  <Bug className="size-5" />
                </TooltipIconButton>
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="New thread"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>

              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16  max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                <>
                  {messages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) =>
                      message.type === "human" ? (
                        <HumanMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                        />
                      ) : (
                        <AssistantMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                          handleRegenerate={handleRegenerate}
                        />
                      ),
                    )}
                  {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                    We need to render it outside of the messages list, since there are no messages to render */}
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
                    <AssistantMessage
                      key="interrupt-msg"
                      message={undefined}
                      isLoading={isLoading}
                      handleRegenerate={handleRegenerate}
                    />
                  )}
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
                </>
              }
              footer={
                <div className="sticky bottom-0 flex flex-col items-center gap-8 bg-white">
                  {!chatStarted && (
                    <div className="flex items-center gap-3">
                      <LangGraphLogoSVG className="h-8 flex-shrink-0" />
                      <h1 className="text-2xl font-semibold tracking-tight">
                        Agent Chat
                      </h1>
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    ref={dropRef}
                    className={cn(
                      "bg-muted relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl shadow-xs transition-all",
                      dragOver
                        ? "border-primary border-2 border-dotted"
                        : "border border-solid",
                    )}
                  >
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <ContentBlocksPreview
                        blocks={contentBlocks}
                        onRemove={removeBlock}
                      />
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onPaste={handlePaste}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder="Type your message..."
                        className="field-sizing-content resize-none border-none bg-transparent p-3.5 pb-0 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div className="flex items-center gap-6 p-2 pt-4">
                        <div>
                          <div className="flex items-center space-x-2">
                            <Switch
                              id="render-tool-calls"
                              checked={hideToolCalls ?? false}
                              onCheckedChange={setHideToolCalls}
                            />
                            <Label
                              htmlFor="render-tool-calls"
                              className="text-sm text-gray-600"
                            >
                              Hide Tool Calls
                            </Label>
                          </div>
                        </div>
                        {singleCapability && !capabilityMenuEnabled && (
                          <div className="flex items-center gap-2">
                            <Switch
                              id={`capability-${singleCapability.name}`}
                              checked={capabilityFlags[singleCapability.name] ?? singleCapability.enabled_by_default}
                              onCheckedChange={(checked: boolean) => setCapabilityFlag(singleCapability.name, checked)}
                            />
                            <Label
                              htmlFor={`capability-${singleCapability.name}`}
                              className="flex items-center gap-2 text-sm text-gray-600"
                            >
                              <SingleCapabilityIcon className="h-4 w-4" />
                              {singleCapability.label}
                            </Label>
                          </div>
                        )}
                        {capabilityMenuEnabled && (
                          <div className="relative">
                            <button
                              type="button"
                              ref={capabilityButtonRef}
                              onClick={() => {
                                setAgentMenuOpen(false);
                                setCapabilityMenuOpen((open) => !open);
                              }}
                              className="flex items-center gap-2 rounded-full border border-transparent bg-muted px-4 py-2 text-sm font-medium transition hover:border-border"
                            >
                              <Globe2 className="h-4 w-4" />
                              <span>Capabilities</span>
                              {activeCapabilityCount > 0 && (
                                <span className="ml-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
                                  {activeCapabilityCount}
                                </span>
                              )}
                              <ChevronDown className="h-4 w-4 text-muted-foreground" />
                            </button>
                            {capabilityMenuOpen && (
                              <div
                                ref={capabilityMenuRef}
                                className="absolute bottom-full left-0 z-50 mb-3 w-80 rounded-2xl border border-border bg-background p-3 shadow-xl"
                              >
                                <div className="flex flex-col gap-3">
                                  {capabilityDefinitions.map((definition) => {
                                    const CapabilityIcon = definition.icon ?? Globe2;
                                    const value = capabilityFlags[definition.name] ?? definition.enabled_by_default;
                                    return (
                                      <div
                                        key={definition.name}
                                        className="flex items-center gap-3"
                                      >
                                        <span className="flex size-9 items-center justify-center rounded-full bg-muted text-primary">
                                          <CapabilityIcon className="h-4 w-4" />
                                        </span>
                                        <div className="flex-1">
                                          <div className="text-sm font-semibold text-foreground">
                                            {definition.label}
                                          </div>
                                          {definition.description && (
                                            <div className="text-xs text-muted-foreground">
                                              {definition.description}
                                            </div>
                                          )}
                                        </div>
                                        <Switch
                                          checked={value}
                                          onCheckedChange={(checked: boolean) => setCapabilityFlag(definition.name, checked)}
                                        />
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                        <Label
                          htmlFor="file-input"
                          className="flex cursor-pointer items-center gap-2"
                        >
                          <Plus className="size-5 text-gray-600" />
                          <span className="text-sm text-gray-600">
                            Upload PDF or Image
                          </span>
                        </Label>
                        <input
                          id="file-input"
                          type="file"
                          onChange={handleFileUpload}
                          multiple
                          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
                          className="hidden"
                        />
                        {stream.isLoading ? (
                          <Button
                            key="stop"
                            onClick={() => stream.stop()}
                            className="ml-auto"
                          >
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            className="ml-auto shadow-md transition-all"
                            disabled={
                              isLoading ||
                              (!input.trim() && contentBlocks.length === 0)
                            }
                          >
                            Send
                          </Button>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button
                onClick={closeArtifact}
                className="cursor-pointer"
              >
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
      <UserMenu />
      <AuditInspector
        open={auditPanelOpen ?? false}
        onClose={() => void setAuditPanelOpen(false)}
        onClear={clearAuditEvents}
        events={auditEvents}
      />
    </div>
  );
}

function UserMenu() {
  const { user, isAdmin, logout, pending } = useAuth();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;

    function handlePointerDown(event: MouseEvent) {
      const target = event.target as Node;
      if (
        menuRef.current &&
        !menuRef.current.contains(target) &&
        triggerRef.current &&
        !triggerRef.current.contains(target)
      ) {
        setOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  if (!user) {
    return null;
  }

  const displayName = user.full_name || user.email;
  const initials = (displayName || "?")
    .trim()
    .charAt(0)
    .toUpperCase() || "?";

  const handleSettings = () => {
    setOpen(false);
    router.push("/admin");
  };

  const handleLogout = () => {
    setOpen(false);
    void logout();
  };

  return (
    <div className="pointer-events-none absolute bottom-6 left-6 z-40 flex flex-col items-start">
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="pointer-events-auto flex size-10 items-center justify-center rounded-full bg-primary text-primary-foreground font-semibold shadow-md transition hover:opacity-90"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {initials}
      </button>
      {open && (
        <div
          ref={menuRef}
          className="pointer-events-auto mt-3 w-60 rounded-2xl border border-border bg-background/95 p-3 text-sm shadow-xl backdrop-blur"
          role="menu"
        >
          <div className="mb-3 border-b pb-2">
            <div className="font-medium text-foreground">{displayName}</div>
            <div className="text-xs text-muted-foreground">
              {user.roles.length > 0 ? user.roles.join(", ") : "member"}
            </div>
          </div>
          <div className="flex flex-col gap-1">
            {isAdmin && (
              <button
                type="button"
                onClick={handleSettings}
                className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left transition hover:bg-muted"
              >
                <Settings className="h-4 w-4" />
                <span>Settings</span>
              </button>
            )}
            <button
              type="button"
              onClick={handleLogout}
              className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left transition hover:bg-muted"
              disabled={pending}
            >
              <LogOut className="h-4 w-4" />
              <span>Sign out</span>
            </button>
            <button
              type="button"
              className="flex w-full cursor-not-allowed items-center gap-2 rounded-xl px-3 py-2 text-left text-muted-foreground"
              disabled
            >
              <Trash2 className="h-4 w-4" />
              <span>Clear all chats (coming soon)</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function AuditInspector(props: {
  open: boolean;
  onClose: () => void;
  onClear: () => void;
  events: StreamAuditEvent[];
}) {
  return (
    <motion.div
      className={cn(
        "pointer-events-auto fixed bottom-0 right-0 z-50 flex h-[60vh] w-full max-w-full flex-col border border-border bg-background shadow-xl md:w-[480px]",
        !props.open && "pointer-events-none",
      )}
      initial={false}
      animate={{ y: props.open ? 0 : "105%" }}
      transition={{ type: "spring", stiffness: 260, damping: 28 }}
    >
      <div className="flex items-center justify-between border-b p-3">
        <div className="text-sm font-semibold">Stream Inspector</div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={props.onClear}
            disabled={props.events.length === 0}
          >
            Clear
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={props.onClose}
          >
            <XIcon className="size-4" />
          </Button>
        </div>
      </div>
      <div className="flex items-center justify-between border-b px-3 py-2 text-xs text-muted-foreground">
        <span>{props.events.length} events</span>
        <span>{props.open ? "Recording" : "Hidden"}</span>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {props.events.length === 0 ? (
          <div className="text-xs text-muted-foreground">
            No events captured yet.
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {props.events.map((event) => {
              const namespace =
                "namespace" in event && Array.isArray(event.namespace)
                  ? event.namespace
                  : undefined;
              const run = "run" in event ? event.run : undefined;
              return (
                <div
                  key={event.id}
                  className="rounded border border-border p-2 text-[11px] leading-snug"
                >
                  <div className="flex items-center justify-between font-semibold uppercase tracking-wide">
                    <span>
                      #{event.sequence} · {event.type}
                    </span>
                    <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                  </div>
                  {namespace && namespace.length > 0 ? (
                    <div className="mt-1 text-muted-foreground">
                      {namespace.join(" / ")}
                    </div>
                  ) : null}
                  {run ? (
                    <div className="mt-1 text-muted-foreground">
                      {run.run_id} · {run.thread_id}
                    </div>
                  ) : null}
                  <pre className="mt-2 whitespace-pre-wrap break-words">
                    {formatAuditPayload(event.payload)}
                  </pre>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}
