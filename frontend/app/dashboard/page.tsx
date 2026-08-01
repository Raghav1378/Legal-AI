"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";


export default function Dashboard() {
  const router = useRouter();
  const { data: session, isPending } = authClient.useSession();

  useEffect(() => {
    if (isPending) return;

    if (!session) {
      router.push("/login");
      return;
    }

    const role = (session.user as any).role || "USER";

    if (role === "ADMIN") {
      router.push("/admin");
    } else if (role === "SUPER_ADMIN") {
      router.push("/super-admin");
    } else {
      router.push("/chat");
    }

  }, [session, isPending, router]);

  return (
    <div className="min-h-screen bg-canvas flex items-center justify-center">
      <div className="text-ink text-lg">Redirecting...</div>
    </div>
  );
}