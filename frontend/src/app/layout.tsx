import type { Metadata } from "next";
import "./globals.css";

import { AuthProvider } from "@/lib/auth-context";

export const metadata: Metadata = {
  title: "Cyrene",
  description: "This will be a LinkedIn journey like none that has come before.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-stone-50 text-stone-900 antialiased">
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
