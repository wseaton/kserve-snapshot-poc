#!/usr/bin/env bash
# Clone an upstream project at its pinned base commit and apply this repo's patches to it.
#
#   ./checkout.sh vllm|modelexpress|dynamo-snapshot [DEST]
#
# DEST defaults to src/<project>. Patches made with git format-patch are applied with git am, so
# each becomes a commit; plain diffs are applied with git apply and left uncommitted.
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
project=${1:?usage: $0 vllm|modelexpress|dynamo-snapshot [DEST]}
dest=${2:-$root/src/$project}
read -r _ repo base _ < <(awk -v p="$project" '$1 == p' "$root/upstream.tsv")
[ -n "${repo:-}" ] || { echo "unknown project $project" >&2; exit 2; }
[ -e "$dest" ] && { echo "$dest exists" >&2; exit 1; }

git init -q "$dest"
git -C "$dest" remote add upstream "$repo"
git -C "$dest" fetch -q --filter=blob:none upstream "$base"
git -C "$dest" checkout -q -b poc "$base"
for patch in "$root/patches/$project"/*.patch; do
  if head -1 "$patch" | grep -q '^From [0-9a-f]\{40\} '; then
    git -C "$dest" am -q "$patch"
  else
    git -C "$dest" apply "$patch"
  fi
done
echo "$dest: $project at ${base:0:12} with $(ls "$root/patches/$project"/*.patch | wc -l | tr -d ' ') patches"
