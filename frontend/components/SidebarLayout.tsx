"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "@/components/services/api";
import { authClient } from "@/lib/auth-client";
import {
  ChevronRight,
  Menu,
  Plus,
  Scale,
  Search,
  Settings,
  Sparkles,
  X,
} from "lucide-react";

export default function SidebarLayout({
  children,
  activeChatId,
}: {
  children: React.ReactNode;
  activeChatId?: string;
}) {
  const router = useRouter();
  const { data: session } = authClient.useSession();
  const role = (session?.user as any)?.role;
  const [chats, setChats] = useState<any[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    void fetchChats();
  }, []);

  useEffect(() => {
    const checkViewport = () => setIsMobile(window.innerWidth < 1024);
    checkViewport();
    window.addEventListener("resize", checkViewport);
    return () => window.removeEventListener("resize", checkViewport);
  }, []);

  const fetchChats = async () => {
    try {
      const res = await apiFetch("/chat/user");
      setChats(res);
    } catch (err) {
      console.error(err);
    }
  };

  const logout = async () => {
    await authClient.signOut();
    router.push("/login");
  };

  const userName = (session?.user as any)?.name || "Legal User";
  const userEmail = (session?.user as any)?.email || "Signed in";
  const initials = useMemo(() => {
    return userName
      .split(" ")
      .map((part: string) => part[0])
      .slice(0, 2)
      .join("")
      .toUpperCase();
  }, [userName]);

  return (
    <div className="flex min-h-screen bg-canvas text-ink">
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed left-4 top-4 z-30 rounded-full border border-line bg-raised/90 p-2 text-ink-muted shadow-lg lg:hidden"
        aria-label="Open sidebar"
      >
        <Menu className="h-5 w-5" />
      </button>

      <aside
        className={`fixed inset-y-0 left-0 z-20 flex w-72 flex-col border-r border-line bg-raised p-4 shadow-2xl shadow-black/30 transition-transform duration-300 lg:static lg:translate-x-0 ${isOpen || !isMobile ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex items-center justify-between px-2 py-2">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-accent-soft p-2 text-accent">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <p className="text-sm font-semibold text-ink">Legal AI Platform</p>
              <p className="text-xs text-ink-muted">Research workspace</p>
            </div>
          </div>
          {isMobile && (
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="rounded-full p-2 text-ink-muted transition hover:bg-hover"
              aria-label="Close sidebar"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <nav className="mt-4 space-y-1">
          <button
            type="button"
            onClick={() => {
              router.push("/chat");
              setIsOpen(false);
            }}
            className="flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left text-sm font-medium text-ink transition hover:bg-hover"
          >
            <span className="flex items-center gap-2.5">
              <Plus className="h-4 w-4 text-accent" />
              New Chat
            </span>
            <ChevronRight className="h-4 w-4 text-ink-faint" />
          </button>

          <button
            type="button"
            className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm text-ink-muted transition hover:bg-hover hover:text-ink"
          >
            <Search className="h-4 w-4" />
            Search
          </button>

          <button
            type="button"
            onClick={() => setShowSettings(true)}
            className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm text-ink-muted transition hover:bg-hover hover:text-ink"
          >
            <Settings className="h-4 w-4" />
            Settings
          </button>

          {(role === "ADMIN" || role === "SUPER_ADMIN") && (
            <button
              type="button"
              onClick={() => {
                router.push("/admin");
                setIsOpen(false);
              }}
              className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm text-ink-muted transition hover:bg-hover hover:text-ink"
            >
              Admin Dashboard
            </button>
          )}

          {role === "SUPER_ADMIN" && (
            <button
              type="button"
              onClick={() => {
                router.push("/super-admin");
                setIsOpen(false);
              }}
              className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm text-ink-muted transition hover:bg-hover hover:text-ink"
            >
              Super Admin
            </button>
          )}
        </nav>

        <div className="mt-6 flex-1 overflow-hidden px-1">
          <div className="mb-2 flex items-center justify-between px-2">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.3em] text-ink-faint">Chat List</h2>
            <span className="text-xs text-ink-faint">{chats.length}</span>
          </div>

          <div className="space-y-1 overflow-y-auto pr-1">
            {chats.map((chat) => (
              <button
                key={chat.id}
                onClick={() => {
                  router.push(`/chat?chatId=${chat.id}`);
                  setIsOpen(false);
                }}
                className={`flex w-full items-center rounded-xl px-3 py-2.5 text-left text-sm transition ${
                  activeChatId === chat.id
                    ? "bg-accent-soft text-ink"
                    : "text-ink-muted hover:bg-hover hover:text-ink"
                }`}
              >
                <span className={`mr-2.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${activeChatId === chat.id ? "bg-accent/20 text-accent" : "bg-hover text-ink-muted"}`}>
                  <Scale className="h-3.5 w-3.5" />
                </span>
                <span className="flex-1 truncate">{chat.title || "Untitled Chat"}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 px-1">
          <button type="button" onClick={() => setShowSettings(true)} className="flex w-full items-center gap-3 rounded-xl px-2 py-2 transition hover:bg-hover">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent-soft text-sm font-semibold text-accent">
              {initials}
            </div>
            <div className="min-w-0 flex-1 text-left">
              <p className="truncate text-sm font-medium text-ink">{userName}</p>
              <p className="truncate text-xs text-ink-muted">{userEmail}</p>
            </div>
          </button>
        </div>
      </aside>

      {isMobile && isOpen && <button type="button" className="fixed inset-0 z-10 bg-black/50" onClick={() => setIsOpen(false)} aria-label="Close sidebar" />}

      {showSettings && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 px-4 py-6">
          <div className="w-full max-w-xl rounded-[28px] border border-line bg-surface p-5 shadow-2xl">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-ink-faint">Settings</p>
                <h3 className="mt-1 text-lg font-semibold text-ink">Account & preferences</h3>
              </div>
              <button type="button" onClick={() => setShowSettings(false)} className="rounded-full p-2 text-ink-muted transition hover:bg-hover">
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-2xl bg-raised p-4">
                <h4 className="font-semibold text-ink">Profile</h4>
                <div className="mt-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-soft text-lg font-semibold text-accent">
                    {initials}
                  </div>
                  <div>
                    <p className="font-medium text-ink">{userName}</p>
                    <p className="text-sm text-ink-muted">{userEmail}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl bg-raised p-4">
                <h4 className="font-semibold text-ink">Appearance</h4>
                <p className="mt-2 text-sm text-ink-muted">Theme changes are coming soon.</p>
              </div>
            </div>

            <div className="mt-4 rounded-2xl bg-raised p-4">
              <h4 className="font-semibold text-ink">Account</h4>
              <p className="mt-2 text-sm text-ink-muted">Use this space to update password support once the backend exposes it.</p>
              <div className="mt-4 flex flex-wrap gap-3">
                <button type="button" className="rounded-xl bg-accent px-3 py-2 text-sm font-medium text-ink transition hover:bg-accent-hover">Change password</button>
                <button type="button" onClick={logout} className="rounded-xl border border-line bg-raised px-3 py-2 text-sm font-medium text-ink-muted transition hover:bg-hover">Logout</button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1">{children}</div>
    </div>
  );
}
