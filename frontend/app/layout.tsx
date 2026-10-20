import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "../components/Sidebar";
import { TopBar } from "../components/TopBar";

export const metadata: Metadata = {
  title: "GlucoTwin - Digital Twin for T2D",
  description: "Decision support, not a medical device",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text-primary flex font-sans antialiased">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopBar />
          <main className="flex-1 p-6 overflow-y-auto max-w-7xl w-full mx-auto space-y-6 animate-page-enter">
            {children}
          </main>
          <footer className="border-t border-border px-6 py-3 text-xs text-text-muted">
            Decision support, not a medical device
          </footer>
        </div>
      </body>
    </html>
  );
}

