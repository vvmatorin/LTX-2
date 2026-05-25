import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["better-sqlite3"],
  typescript: {
    ignoreBuildErrors: false,
  },
  devIndicators: false,
};

export default nextConfig;
