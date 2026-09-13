#!/usr/bin/env bash
# THIS WILL SET UP THE DATABASE FOR THE APPLICATION USING schema.sql
set -euo pipefail

if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL is not set" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

psql "$DATABASE_URL" -f "$SCRIPT_DIR/schema.sql"
