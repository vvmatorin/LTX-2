#!/usr/bin/env bash
# Install dependencies and launch the LTX-2 training UI. Pass --dev for hot-reload.
set -euo pipefail
cd "$(dirname "$0")"

export PORT="${PORT:-8675}"
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export TENSORBOARD_HOST="${TENSORBOARD_HOST:-127.0.0.1}"
# Must match the rewrite in ui/next.config.ts and api/tensorboard/route.ts.
export TENSORBOARD_PATH_PREFIX="${TENSORBOARD_PATH_PREFIX:-/tensorboard}"

NODE_MIN_MAJOR=20
NODE_VERSION=22

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

node_ok() {
    command -v node &>/dev/null && command -v npm &>/dev/null &&
        (( $(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0) >= NODE_MIN_MAJOR ))
}

# nvm.sh does not survive `set -eu`, so relax the shell around it.
load_nvm() {
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    [[ -s "$NVM_DIR/nvm.sh" ]] || return 1
    set +eu
    . "$NVM_DIR/nvm.sh"
    set -eu
}

# Python environment
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
info "Syncing Python dependencies..."
uv sync
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Node.js
if ! node_ok; then
    info "Installing Node ${NODE_VERSION} via nvm..."
    if ! load_nvm; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        load_nvm || die "nvm install failed; install Node ${NODE_MIN_MAJOR}+ manually and retry."
    fi
    set +eu
    nvm install "$NODE_VERSION" && nvm alias default "$NODE_VERSION" && nvm use "$NODE_VERSION"
    set -eu
    node_ok || die "Node ${NODE_MIN_MAJOR}+ still unavailable; install it manually and retry."
fi
info "Using Node $(node -v) with npm $(npm -v)"

# UI dependencies
cd ui
npm install

# better-sqlite3 is native and tied to one Node ABI; `npm install` won't rebuild
# it when the Node version changes, so stamp the ABI and rebuild on mismatch.
NODE_ABI="$(node -p 'process.versions.modules')"
ABI_STAMP="node_modules/.ltx-node-abi"
if [[ "$(cat "$ABI_STAMP" 2>/dev/null)" != "$NODE_ABI" ]]; then
    info "Rebuilding native modules for Node $(node -v)..."
    npm rebuild
    echo "$NODE_ABI" > "$ABI_STAMP"
fi

# Launch
info "Launching UI on http://localhost:${PORT} (TensorBoard at ${TENSORBOARD_PATH_PREFIX}/)"
if [[ "${1:-}" == "--dev" ]]; then
    exec npm run dev
else
    npm run build
    exec npm run start
fi
