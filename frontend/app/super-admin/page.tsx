"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/components/services/api";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";
import SidebarLayout from "@/components/SidebarLayout";
import ProtectedLayout from "@/components/ProtectedLayout";

export default function SuperAdminPage() {
  const router = useRouter();
  const [sources, setSources] = useState<any[]>([]);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);

  useEffect(() => {
    fetchData();
  }, []);


  const fetchData = async () => {
    const sourcesRes = await apiFetch("/repository-sources");
    const auditRes = await apiFetch("/admin/audit-logs");

    setSources(sourcesRes);
    setAuditLogs(auditRes);
  };

  const toggleSource = async (id: string, enabled: boolean) => {
    await apiFetch(`/repository-sources/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled: !enabled }),
    });

    fetchData();
  };

  return (
    <ProtectedLayout allowedRoles={["SUPER_ADMIN"]}>
      <SidebarLayout>
        <div className="min-h-screen bg-canvas px-6 py-10">
      <div className="max-w-6xl mx-auto space-y-10">

        <div>
          <h1 className="text-3xl font-semibold text-ink">
            Super Admin Panel
          </h1>
          <p className="text-ink-faint text-sm mt-1">
            System configuration & audit monitoring
          </p>
        </div>

        <div className="bg-raised border border-line rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-ink mb-6">
            Research Sources
          </h2>

          <div className="space-y-4">
            {sources.map((source) => (
              <div
                key={source.id}
                className="flex items-center justify-between border-b border-line pb-3"
              >
                <span className="text-ink">
                  {source.name}
                </span>

                <button
                  onClick={() =>
                    toggleSource(source.id, source.enabled)
                  }
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
                    source.enabled
                      ? "bg-green-500/20 text-green-400 hover:bg-green-500/30"
                      : "bg-hover text-ink-muted hover:bg-press"
                  }`}
                >
                  {source.enabled ? "Enabled" : "Disabled"}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-raised border border-line rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-ink mb-6">
            Audit Logs
          </h2>

          <div className="space-y-4">
            {auditLogs.map((log) => (
              <div
                key={log.id}
                className="border-b border-line pb-3"
              >
                <p className="text-ink">
                  <span className="font-medium">
                    {log.action}
                  </span>{" "}
                  — {log.user?.email || "System"}
                </p>
                <p className="text-xs text-ink-faint mt-1">
                  {new Date(log.createdAt).toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
      </SidebarLayout>
    </ProtectedLayout>
  );
}