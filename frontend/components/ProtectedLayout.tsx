"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";

export default function ProtectedLayout({
  children,
  allowedRoles,
}: {
  children: React.ReactNode;
  allowedRoles: string[];
}) {
  const router = useRouter();
  const { data: session, isPending: loading } = authClient.useSession();
  const [authorized, setAuthorized] = useState(false);

  useEffect(() => {
    if (loading) return;

    if (!session) {
      console.log("[ProtectedLayout] No session found, redirecting to /login");
      router.push("/login");
      return;
    }

    const userRole = ((session.user as any).role || "USER") as string;
    console.log("[ProtectedLayout] Current Role:", userRole, "Allowed:", allowedRoles);

    if (!allowedRoles.includes(userRole)) {
      console.log("[ProtectedLayout] Role not authorized, redirecting to /dashboard");
      router.push("/dashboard");
      return;
    }

    setAuthorized(true);
  }, [session, loading, allowedRoles, router]);

  if (loading || !authorized) {
    return (
      <div className="min-h-screen bg-canvas flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="text-ink text-lg animate-pulse">Loading session...</div>
          <div className="text-ink-faint text-xs text-center px-6 max-w-xs">
            Verifying your access credentials. This should only take a moment.
          </div>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}