#!/usr/bin/env bash
# Регенерация золотых эталонов для C#-порта. Эталон — ВСЕГДА вывод этих
# экспортёров: править файлы в пакете руками нельзя, иначе паритет проверяет
# сам себя.
#   scripts/goldens.sh [OUT_DIR]
# OUT_DIR по умолчанию ../dioma-engine/Tests/Editor/Data (соседний репо).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

OUT="${1:-$ROOT/../dioma-engine/Tests/Editor/Data}"
if [[ ! -d "$(dirname "$OUT")" ]]; then
  echo "нет каталога $(dirname "$OUT") — репо dioma-engine должен лежать рядом с dioma" >&2
  exit 1
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
echo "эталоны → $OUT"
echo

# 4 файла: dioma_{connectome,golden}.json + dioma_c_{connectome,golden}.json
cargo run --release --quiet --bin export_connectome -- "$OUT" both
# dioma_bio_golden.json
cargo run --release --quiet --bin export_bio -- "$OUT"
# dioma_reference.taas + dioma_reference_taas_expect.json
cargo run --release --quiet --bin export_taas -- "$OUT"
# dioma_skinner_{connectome,golden}.json
cargo run --release --quiet --bin export_learning -- "$OUT"

echo
echo "── состояние пакета после регенерации ──"
if git -C "$OUT" rev-parse --git-dir >/dev/null 2>&1; then
  changed="$(git -C "$OUT" status --short -- . | grep -v '\.meta$' || true)"
  if [[ -z "$changed" ]]; then
    echo "эталоны байт-в-байт совпали с закоммиченными — расхождений нет"
  else
    echo "эталоны ИЗМЕНИЛИСЬ (значит поменялась референсная модель):"
    printf '%s\n' "$changed"
    echo
    echo "прогони проверки C#-порта, прежде чем коммитить:"
    echo "  (cd $OUT/../../.. && scripts/verify.sh)"
  fi
else
  echo "$OUT не под git — сверить не с чем"
fi
