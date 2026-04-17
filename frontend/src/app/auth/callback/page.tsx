"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function AuthCallback() {
  const router = useRouter();

  useEffect(() => {
    async function handleCallback() {
      try {
        const { supabase } = await import("@/lib/supabase");
        const { data, error } = await supabase.auth.exchangeCodeForSession(
          window.location.href
        );

        if (error) {
          console.error("Auth error:", error);
          router.push("/");
          return;
        }

        if (data.session) {
          localStorage.setItem("access_token", data.session.access_token);
          router.push("/home");
        }
      } catch (e) {
        console.error("Callback error:", e);
        router.push("/");
      }
    }

    handleCallback();
  }, [router]);

  return (
    <div className="flex min-h-screen items-center justify-center">
      <p className="text-stone-500">Signing in...</p>
    </div>
  );
}
