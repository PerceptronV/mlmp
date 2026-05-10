#!/usr/bin/env bash
# Download an OSF project's osfstorage as a zip and extract it into src/data/.
#
# Usage:
#   ./scripts/setup.sh                       # downloads gq2hj into src/data/osf_gq2hj
#   ./scripts/setup.sh <project_id>          # downloads <project_id> into src/data/osf_<project_id>
#   ./scripts/setup.sh <project_id> <dir>    # downloads <project_id> into src/data/<dir>

set -euo pipefail

PROJECT_ID="${1:-gq2hj}"
OUTPUT_DIR="${2:-osf_${PROJECT_ID}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/src/data/${OUTPUT_DIR}"
ZIP_PATH="${TARGET_DIR}.zip"
URL="https://files.osf.io/v1/resources/${PROJECT_ID}/providers/osfstorage/?zip="

if [[ -d "$TARGET_DIR" && -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty directory: $TARGET_DIR" >&2
  echo "Delete it first if you want to redownload." >&2
  exit 1
fi

for cmd in curl unzip; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

echo "Downloading OSF project '${PROJECT_ID}' from ${URL}"
echo "  -> ${ZIP_PATH}"
curl --fail --location --progress-bar -o "$ZIP_PATH" "$URL"

echo "Extracting into ${TARGET_DIR}/"
mkdir -p "$TARGET_DIR"
unzip -q "$ZIP_PATH" -d "$TARGET_DIR"

rm -f "$ZIP_PATH"
echo "Done. Files are in ${TARGET_DIR}/"
