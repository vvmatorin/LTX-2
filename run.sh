#!/usr/bin/env bash
set -euo pipefail

# ── LTX-2: install dependencies and launch the training UI ─────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration (single source of truth) ─────────────────────────────────
export PORT="${PORT:-8675}"
# TensorBoard process. Bound to loopback and reverse-proxied through Next.js
# so users only need to forward one port.
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export TENSORBOARD_HOST="${TENSORBOARD_HOST:-127.0.0.1}"
# Sub-path under which TensorBoard is proxied. Must match the rewrite in
# ui/next.config.ts and the --path_prefix flag in api/tensorboard/route.ts.
export TENSORBOARD_PATH_PREFIX="${TENSORBOARD_PATH_PREFIX:-/tensorboard}"

# ── Helpers ────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

# ── 1. Python environment (uv) ─────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    command -v curl &>/dev/null || die "curl not found; install uv manually (https://astral.sh/uv) and retry."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

info "Syncing Python dependencies (uv sync)..."
uv sync

export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# ── 2. Node.js / npm ───────────────────────────────────────────────────────
NODE_MIN_MAJOR=20
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-22}"
NVM_INSTALLER="https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh"

node_ok() {
    command -v node &>/dev/null || return 1
    command -v npm  &>/dev/null || return 1
    (( "$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)" >= NODE_MIN_MAJOR ))
}

# nvm.sh is not written to survive `set -euo pipefail`, hence the toggling.
load_nvm() {
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    [[ -s "$NVM_DIR/nvm.sh" ]] || return 1
    set +eu
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"
    set -eu
}

if ! node_ok; then
    if command -v node &>/dev/null; then
        info "Node $(node -v) is too old (need >= ${NODE_MIN_MAJOR}); installing Node ${NODE_INSTALL_VERSION} via nvm..."
    else
        info "Node.js not found; installing Node ${NODE_INSTALL_VERSION} via nvm..."
    fi

    if ! load_nvm; then
        command -v curl &>/dev/null || die "curl not found; install Node ${NODE_INSTALL_VERSION}+ manually and retry."
        curl -o- "$NVM_INSTALLER" | bash || die "Failed to download nvm from ${NVM_INSTALLER}."
        load_nvm || die "nvm install did not complete. Install Node ${NODE_INSTALL_VERSION}+ manually and retry."
    fi

    set +eu
    nvm install "$NODE_INSTALL_VERSION"
    nvm alias default "$NODE_INSTALL_VERSION"
    nvm use "$NODE_INSTALL_VERSION"
    set -eu

    node_ok || die "Node is still $(node -v 2>/dev/null || echo 'missing') after installing via nvm.
       Install Node ${NODE_INSTALL_VERSION}+ manually and retry."
fi

info "Using Node $(node -v) with npm $(npm -v)"

# ── 3. Node dependencies ───────────────────────────────────────────────────
cd "$SCRIPT_DIR/ui"
npm install

# Native modules (better-sqlite3) are compiled against one Node ABI and fail to
# load under another. `npm install` will not rebuild them on its own once the
# package is present at the locked version, so stamp the ABI we built for and
# rebuild whenever it changes — this repo may be shared between machines.
NODE_ABI="$(node -p 'process.versions.modules')"
ABI_STAMP="node_modules/.ltx-node-abi"
if [[ "$(cat "$ABI_STAMP" 2>/dev/null || true)" != "$NODE_ABI" ]]; then
    info "Building native modules for Node $(node -v) (ABI ${NODE_ABI})..."
    npm rebuild
    printf '%s\n' "$NODE_ABI" > "$ABI_STAMP"
fi

# ── 4. Launch UI (Next.js + worker via concurrently) ───────────────────────
info "Launching UI on http://localhost:${PORT}  (TensorBoard at ${TENSORBOARD_PATH_PREFIX}/)"

if [[ "${1:-}" == "--dev" ]]; then
    info "Running in development mode (hot-reload)..."
    exec npm run dev
else
    info "Running production build..."
    npm run build
    exec npm run start
fi
