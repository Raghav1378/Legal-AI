"use client";

import { useState, useEffect, Suspense, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { apiFetch } from "@/components/services/api";
import StructuredResponse from "@/components/StructuredResponse";
import SidebarLayout from "@/components/SidebarLayout";
import ProtectedLayout from "@/components/ProtectedLayout";
import {
  Bell,
  Bookmark,
  BrainCircuit,
  LoaderCircle,
  MessageSquareText,
  Paperclip,
  Scale,
  SearchCheck,
  SendHorizontal,
  Sparkles,
} from "lucide-react";

const capabilityPills = [
  { label: "Case Research", icon: SearchCheck },
  { label: "Draft Document", icon: Sparkles },
  { label: "Summarize Filing", icon: Scale },
  { label: "Ask a Question", icon: BrainCircuit },
];

function ChatPageContent() {
  const searchParams = useSearchParams();
  const chatIdFromUrl = searchParams.get("chatId");

  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [activeChatId, setActiveChatId] = useState<string | undefined>();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (chatIdFromUrl) {
      loadChat(chatIdFromUrl);
    } else {
      setActiveChatId(undefined);
      setResponse(null);
    }
  }, [chatIdFromUrl]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [query]);

  const handleSubmit = async () => {
    if (!query.trim()) return;

    const submittedQuery = query.trim();
    setLoading(true);
    try {
      const res = await apiFetch("/chat", {
        method: "POST",
        body: JSON.stringify({
          query: submittedQuery,
          chatId: activeChatId,
        }),
      });

      setResponse({
        ...res,
        messageId: res.messageId,
        isBookmarked: res.isBookmarked,
        userQuery: submittedQuery,
      });

      setActiveChatId(res.chatId);
      setQuery("");
    } catch (err) {
      console.error(err);
      alert("Error generating response");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event: any) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  };

  const loadChat = async (chatId: string) => {
    try {
      const chat = await apiFetch(`/chat/${chatId}`);

      if (!chat?.messages?.length) {
        setResponse(null);
        return;
      }

      const assistantMessages = chat.messages.filter(
        (m: any) => m.role === "ASSISTANT"
      );

      if (!assistantMessages.length) {
        setResponse(null);
        return;
      }

      const latestAssistant =
        assistantMessages[assistantMessages.length - 1];

      setResponse({
        ...latestAssistant.structuredResponse,
        messageId: latestAssistant.id,
        isBookmarked: latestAssistant.isBookmarked,
      });

      setActiveChatId(chatId);
    } catch (err) {
      console.error("Load chat error:", err);
    }
  };

  return (
    <ProtectedLayout allowedRoles={["USER", "ADMIN", "SUPER_ADMIN"]}>
      <SidebarLayout activeChatId={activeChatId}>
        <div className="flex min-h-screen flex-col bg-canvas text-ink">
          <header className="border-b border-line bg-surface/70 px-4 py-4 backdrop-blur sm:px-6">
            <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.35em] text-accent">
                  AI Legal Research Assistant
                </p>
                <h1 className="mt-1 text-lg font-semibold text-ink">
                  Structured analysis for legal questions
                </h1>
              </div>
              <div className="flex items-center gap-2">
                <button type="button" className="rounded-full border border-line bg-raised p-2 text-ink-muted transition hover:bg-hover hover:text-ink" aria-label="Notifications">
                  <Bell className="h-4 w-4" />
                </button>
                <button type="button" className="rounded-full border border-line bg-raised p-2 text-ink-muted transition hover:bg-hover hover:text-ink" aria-label="Bookmark">
                  <Bookmark className="h-4 w-4" />
                </button>
                <button type="button" className="hidden rounded-full border border-line bg-raised px-3 py-2 text-sm font-medium text-ink-muted transition hover:bg-hover hover:text-ink sm:inline-flex">
                  Share
                </button>
                {/* TODO: optional sources drawer */}
              </div>
            </div>
          </header>

          <main className="flex-1 overflow-hidden px-4 py-6 sm:px-6 lg:px-8">
            <div className="mx-auto flex h-full max-w-7xl">
              <section className="flex min-h-0 flex-1 flex-col">
                <div className="flex-1 overflow-y-auto px-1 sm:px-2">
                  {!response && !loading && (
                    <div className="mx-auto flex min-h-full max-w-3xl flex-col items-center justify-center px-2 py-10 text-center">
                      <div className="rounded-2xl bg-accent-soft p-4 text-accent">
                        <SearchCheck className="h-8 w-8" />
                      </div>
                      <h2 className="mt-6 text-2xl font-semibold text-ink">
                        Ask anything about law.
                      </h2>
                      <p className="mt-3 max-w-2xl text-sm leading-7 text-ink-muted sm:text-base">
                        Get structured legal analysis with case references, conflict screening, and confidence scoring — all in one workspace.
                      </p>

                      <div className="mt-8 grid w-full gap-3 sm:grid-cols-2">
                        {capabilityPills.map((pill) => {
                          const Icon = pill.icon;
                          return (
                            <button
                              key={pill.label}
                              type="button"
                              onClick={() => {
                                setQuery(pill.label);
                                textareaRef.current?.focus();
                              }}
                              className="group rounded-2xl bg-raised px-4 py-3 text-left text-sm text-ink-muted transition duration-200 hover:-translate-y-0.5 hover:bg-hover hover:text-ink"
                            >
                              <div className="mb-2 inline-flex rounded-full bg-hover p-2 text-ink-muted transition group-hover:bg-accent-soft group-hover:text-accent">
                                <Icon className="h-4 w-4" />
                              </div>
                              <div className="font-medium text-ink">{pill.label}</div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {loading && !response && (
                    <div className="mx-auto max-w-3xl rounded-3xl bg-raised p-6">
                      <div className="flex items-center gap-3">
                        <div className="rounded-full bg-accent-soft p-2 text-accent">
                          <LoaderCircle className="h-5 w-5 animate-spin" />
                        </div>
                        <div>
                          <p className="font-medium text-ink">Analyzing your legal request</p>
                          <p className="text-sm text-ink-muted">
                            Gathering relevant authorities, conflict signals, and next steps.
                          </p>
                        </div>
                      </div>

                      <div className="mt-6 space-y-3">
                        {[1, 2, 3].map((item) => (
                          <div key={item} className="h-4 animate-pulse rounded-full bg-hover" />
                        ))}
                      </div>
                    </div>
                  )}

                  {response && (
                    <div className="mx-auto flex max-w-3xl flex-col gap-4">
                      {response.userQuery && (
                        <div className="flex justify-end">
                          <div className="max-w-[82%] rounded-2xl bg-raised px-4 py-3 text-sm font-medium text-ink shadow-sm">
                            {response.userQuery}
                          </div>
                        </div>
                      )}

                      <div className="rounded-3xl bg-raised p-4 sm:p-6">
                        <StructuredResponse data={response} />
                      </div>
                    </div>
                  )}
                </div>

                <footer className="px-1 py-4 sm:px-2">
                  <div className="mx-auto max-w-3xl">
                    <div className="rounded-3xl border border-line bg-surface p-2 shadow-lg shadow-black/20">
                      <textarea
                        ref={textareaRef}
                        className="max-h-44 min-h-14.5 w-full resize-none bg-transparent px-3 py-3 text-sm text-ink outline-none placeholder:text-ink-faint"
                        rows={1}
                        placeholder="Ask your legal question..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                      />

                      <div className="mt-2 flex items-center justify-between px-2 pb-1">
                        <div className="flex items-center gap-2 text-xs text-ink-faint">
                          {loading ? (
                            <>
                              <LoaderCircle className="h-3.5 w-3.5 animate-spin text-accent" />
                              Analyzing your request…
                            </>
                          ) : (
                            <>
                              <MessageSquareText className="h-3.5 w-3.5" />
                              Press Enter to send · Shift + Enter for a new line
                            </>
                          )}
                        </div>

                        <div className="flex items-center gap-2">
                          <button type="button" className="flex h-10 w-10 items-center justify-center rounded-full bg-hover text-ink-muted transition hover:bg-press hover:text-ink" aria-label="Attach">
                            <Paperclip className="h-4 w-4" />
                          </button>
                          <button type="button" onClick={handleSubmit} disabled={loading || !query.trim()} className="flex h-10 w-10 items-center justify-center rounded-full bg-accent text-ink shadow-lg shadow-accent/20 transition duration-200 hover:bg-accent-hover hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-60">
                            {loading ? (
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                            ) : (
                              <SendHorizontal className="h-4 w-4" />
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </footer>
              </section>
            </div>
          </main>
        </div>
      </SidebarLayout>
    </ProtectedLayout>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-canvas text-ink-muted">Loading chat…</div>}>
      <ChatPageContent />
    </Suspense>
  );
}
