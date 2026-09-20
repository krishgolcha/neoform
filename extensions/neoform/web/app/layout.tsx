import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEOFORM — Evolutionary post-training",
  description: "An open-source evolution lab powered by Tinker.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

