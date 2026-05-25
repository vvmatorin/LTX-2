import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/Providers";
import { Sidebar } from "@/components/Sidebar";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "LTX Trainer",
  description: "Training UI for LTX-2 video generation models",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistMono.variable} dark h-full antialiased`}
    >
      <body className="h-full bg-background text-foreground">
        <Providers>
          <div className="relative flex h-full overflow-hidden">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_78%_8%,oklch(0.7_0.19_235_/_0.18),transparent_35%),radial-gradient(circle_at_0%_0%,oklch(0.68_0.2_294_/_0.2),transparent_30%)]" />
            <Sidebar />
            <main className="relative z-10 flex-1 overflow-auto p-4 md:p-6">
              <div className="app-shell min-h-full rounded-3xl p-3 md:p-4">{children}</div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
