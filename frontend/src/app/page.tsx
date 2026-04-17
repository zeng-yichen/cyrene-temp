"use client";

import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  return (
    <div className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-md space-y-8 rounded-2xl bg-white p-10 shadow-sm">
        <div className="text-center">
          <h1 className="text-4xl font-semibold tracking-tight text-stone-900">Cyrene</h1>
          <p className="mt-3 text-sm italic text-stone-400">
            This will be a LinkedIn journey like none that has come before.
          </p>
        </div>

        <button
          onClick={() => router.push("/home")}
          className="w-full rounded-lg bg-stone-900 px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-stone-800"
        >
          Sign in with Google
        </button>
      </div>
    </div>
  );
}
