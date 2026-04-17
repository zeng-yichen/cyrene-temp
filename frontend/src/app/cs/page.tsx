"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { csApi } from "@/lib/api";

interface Client {
  id: string;
  first_name: string;
  last_name: string;
  email: string;
  company_id: string;
  title: string | null;
}

export default function CSDashboard() {
  const [clients, setClients] = useState<Client[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    csApi
      .listClients()
      .then((data) => setClients(data.clients))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <h1 className="text-2xl font-semibold">Customer Success</h1>
      <p className="mt-1 text-sm text-stone-500">Client health and content pipeline</p>

      <div className="mt-8 overflow-hidden rounded-xl border border-stone-200 bg-white shadow-sm">
        <table className="w-full">
          <thead>
            <tr className="border-b border-stone-100 bg-stone-50">
              <th className="px-4 py-3 text-left text-xs font-medium text-stone-500">Name</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-stone-500">Email</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-stone-500">Title</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-stone-500">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-stone-400">
                  Loading...
                </td>
              </tr>
            ) : clients.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-stone-400">
                  No clients found
                </td>
              </tr>
            ) : (
              clients.map((client) => (
                <tr key={client.id} className="border-b border-stone-50 hover:bg-stone-50">
                  <td className="px-4 py-3 text-sm font-medium">
                    {client.first_name} {client.last_name}
                  </td>
                  <td className="px-4 py-3 text-sm text-stone-500">{client.email}</td>
                  <td className="px-4 py-3 text-sm text-stone-500">{client.title || "—"}</td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/cs/${client.id}`}
                      className="text-sm text-stone-600 hover:text-stone-900"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
