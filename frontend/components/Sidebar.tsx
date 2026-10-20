"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Users, Cpu, FlaskConical, FileText, ShieldCheck } from "lucide-react";
import { useGlucoTwinStore } from "../lib/store";

export const Sidebar: React.FC = () => {
  const pathname = usePathname();
  const { selectedPatient } = useGlucoTwinStore();
  const currentPid = selectedPatient?.id || "IND_PAT_001";

  const navItems = [
    { label: "Patients", href: "/patients", icon: Users },
    { label: "Twin", href: `/twin/${currentPid}`, icon: Cpu },
    { label: "What-if", href: "/what-if", icon: FlaskConical },
    { label: "Summary", href: `/summary/${currentPid}`, icon: FileText },
    { label: "Evidence", href: "/evidence", icon: ShieldCheck },
  ];

  return (
    <aside className="w-64 bg-surface border-r border-border hidden md:flex flex-col justify-between p-4">
      <div className="space-y-6">
        <Link href="/" className="block px-2 py-1">
          <div className="flex items-baseline gap-1.5">
            <span className="text-lg font-bold text-text-primary tracking-tight">GlucoTwin</span>
            <span className="text-xs text-text-muted font-normal">by ZIZ</span>
          </div>
        </Link>

        <nav className="space-y-1">
          {navItems.map((item) => {
            const isActive =
              pathname === item.href ||
              (item.href.startsWith("/twin") && pathname.startsWith("/twin")) ||
              (item.href.startsWith("/summary") && pathname.startsWith("/summary"));
            const Icon = item.icon;
            return (
              <Link
                key={item.label}
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-brand/10 text-brand border border-brand/30 ring-1 ring-inset ring-brand/20 font-semibold"
                    : "text-text-secondary hover:text-text-primary hover:bg-surface-muted"
                }`}
              >
                <Icon size={16} strokeWidth={1.75} className="shrink-0" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="border-t border-border pt-3 px-2">
        <p className="text-[11px] text-text-muted">
          Decision support, not a medical device
        </p>
      </div>
    </aside>
  );
};

