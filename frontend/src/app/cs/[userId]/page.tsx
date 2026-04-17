"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { csApi } from "@/lib/api";

export default function ClientDetail() {
  const params = useParams();
  const userId = params.userId as string;
  const [data, setData] = useState<{ user: any; posts: any[] } | null>(null);

  useEffect(() => {
    csApi.getClient(userId).then(setData).catch(console.error);
  }, [userId]);

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-stone-400">Loading...</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <h1 className="text-2xl font-semibold">
        {data.user?.first_name} {data.user?.last_name}
      </h1>
      <p className="text-sm text-stone-500">{data.user?.email}</p>

      <div className="mt-8 grid grid-cols-3 gap-6">
        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm">
          <h3 className="text-xs font-medium text-stone-500">Total Posts</h3>
          <p className="mt-1 text-3xl font-semibold">{data.posts.length}</p>
        </div>
        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm">
          <h3 className="text-xs font-medium text-stone-500">Published</h3>
          <p className="mt-1 text-3xl font-semibold">
            {data.posts.filter((p: any) => p.status === "posted").length}
          </p>
        </div>
        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm">
          <h3 className="text-xs font-medium text-stone-500">In Review</h3>
          <p className="mt-1 text-3xl font-semibold">
            {data.posts.filter((p: any) => p.status === "review").length}
          </p>
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-lg font-medium">Recent Posts</h2>
        <div className="mt-4 space-y-3">
          {data.posts.slice(0, 10).map((post: any) => (
            <div
              key={post.id}
              className="rounded-lg border border-stone-200 bg-white p-4 shadow-sm"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{post.hook || "Untitled"}</span>
                <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">
                  {post.status}
                </span>
              </div>
              <p className="mt-1 text-xs text-stone-400">{post.post_date || post.created_at}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
