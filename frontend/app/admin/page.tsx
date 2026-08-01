"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/components/services/api";
import { authClient } from "@/lib/auth-client";

import SidebarLayout from "@/components/SidebarLayout";
import ProtectedLayout from "@/components/ProtectedLayout";

export default function AdminPage() {
  const router = useRouter();
  const [queries, setQueries] = useState<any[]>([]);
  const [analytics, setAnalytics] = useState<any>(null);

  useEffect(() => {
    fetchData();
  }, []);


  const fetchData = async () => {
    try {
      const queriesRes = await apiFetch("/admin/queries");
      const analyticsRes = await apiFetch("/admin/analytics");

      setQueries(queriesRes);
      setAnalytics(analyticsRes);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <ProtectedLayout allowedRoles={["ADMIN", "SUPER_ADMIN"]}>
      <SidebarLayout>
        <div className="min-h-screen bg-canvas px-6 py-10">
      <div className="max-w-6xl mx-auto space-y-10">

        <div>
          <h1 className="text-3xl font-semibold text-ink">
            Admin Dashboard
          </h1>
          <p className="text-ink-faint text-sm mt-1">
            System analytics and user query insights
          </p>
        </div>

        {analytics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <Card title="Total Queries" value={analytics.totalQueries} />
            <Card
              title="Avg Confidence"
              value={`${Math.round(analytics.avgConfidence || 0)}%`}
            />
            <Card title="Conflicts" value={analytics.conflictCount} />
            <Card title="Agent Failures" value={analytics.agentFailures} />
          </div>
        )}

        <div className="bg-raised border border-line rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-ink mb-6">
            All Queries
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full text-sm text-ink-muted">
              <thead>
                <tr className="border-b border-line text-ink-faint">
                  <th className="text-left py-3">User</th>
                  <th className="text-left py-3">Confidence</th>
                  <th className="text-left py-3">Conflict</th>
                  <th className="text-left py-3">Created</th>
                </tr>
              </thead>
              <tbody>
                {queries.map((chat) =>
                  chat.messages.map((msg: any, index: number) => (
                    <tr
                      key={index}
                      className="border-b border-line hover:bg-hover transition"
                    >
                      <td className="py-3 text-ink">
                        {chat.user.email}
                      </td>
                      <td className="py-3">
                        {msg.confidenceScore}%
                      </td>
                      <td className="py-3">
                        {msg.conflictsDetected ? (
                          <span className="text-red-400 font-medium">
                            Yes
                          </span>
                        ) : (
                          <span className="text-green-400">No</span>
                        )}
                      </td>
                      <td className="py-3">
                        {new Date(msg.createdAt).toLocaleString()}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
      </SidebarLayout>
    </ProtectedLayout>
  );
}

function Card({ title, value }: { title: string; value: any }) {
  return (
    <div className="bg-raised border border-line rounded-2xl p-6">
      <p className="text-ink-faint text-sm mb-2">{title}</p>
      <p className="text-2xl font-semibold text-ink">
        {value}
      </p>
    </div>
  );
}