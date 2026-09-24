#!/usr/bin/env bash
# Download the trained model checkpoints from the GitHub Release and unpack them
# into this repo (runs_7251/<config>/fold_N/, runs_pest/<config>/fold_N/).
#
#   bash scripts/download_checkpoints.sh
#
# Needs the GitHub CLI (gh), logged in or with GH_TOKEN set. About 14 GB.
# Safe to re-run: archives already unpacked are skipped. Checksums are verified at the end.
set -euo pipefail
REPO="DinaKhasanova/physchem-xai"
TAG="checkpoints-v1"

cd "$(git rev-parse --show-toplevel)"
TMP="${TMPDIR:-/tmp}/physchem_ckpt_download"
mkdir -p "$TMP"

gh release download "$TAG" -R "$REPO" -p ckpt_md5.txt -D "$TMP" --clobber
names=$(gh release view "$TAG" -R "$REPO" --json assets -q '.assets[].name' | grep '\.tar$')
total=$(echo "$names" | wc -l | tr -d ' ')

i=0
for name in $names; do
  i=$((i + 1))
  if [ -f "$TMP/$name.done" ]; then echo "[$i/$total] $name already unpacked, skipping"; continue; fi
  echo "[$i/$total] downloading $name"
  gh release download "$TAG" -R "$REPO" -p "$name" -D "$TMP" --clobber
  tar -xf "$TMP/$name"
  rm -f "$TMP/$name"
  touch "$TMP/$name.done"
done

echo "verifying checksums..."
md5_of() { if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
n=0; bad=0
while read -r hash path; do
  n=$((n + 1))
  if [ ! -f "$path" ] || [ "$(md5_of "$path")" != "$hash" ]; then echo "  MISMATCH: $path"; bad=$((bad + 1)); fi
done < "$TMP/ckpt_md5.txt"

if [ "$bad" -eq 0 ]; then
  echo "All $n checkpoints downloaded and verified."
  rm -rf "$TMP"
else
  echo "$bad of $n checkpoints failed. Delete $TMP/*.done and run again."
  exit 1
fi
