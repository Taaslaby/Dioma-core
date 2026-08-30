#!/usr/bin/env bash
# Проверки Rust-стороны dioma.
#   scripts/verify.sh          → cargo check + тесты крейта dioma-engine (быстро)
#   scripts/verify.sh all      → плюс тесты SiMuse (свой воркспейс, долго)
#   scripts/verify.sh fmt      → cargo fmt --check (только показать)
#   scripts/verify.sh lint     → clippy по крейтам dioma
# fmt/lint вынесены отдельно намеренно: репозиторий писался без rustfmt,
# массовое переформатирование — отдельное решение, не побочный эффект проверки.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-quick}"

case "$MODE" in
  fmt)
    # SiMuse — отдельный воркспейс, у него свой fmt.
    cargo fmt --all -- --check
    ;;
  lint)
    cargo clippy --workspace --all-targets -- -D warnings
    ;;
  quick|all)
    echo "── cargo check (dioma + path-зависимости) ──"
    cargo check --all-targets

    echo
    echo "── тесты crates/dioma-engine (220 штук) ──"
    # Крейт не член воркспейса (у SiMuse свой), поэтому `-p dioma-engine` из
    # корня не работает — заходим внутрь. --release обязателен: в debug мозг
    # считается на порядок медленнее и тесты со сном/голоданием не укладываются
    # в разумное время (145с в release против 9+ минут в debug).
    (cd crates/dioma-engine && cargo test --lib --release)

    if [[ "$MODE" == "all" ]]; then
      echo
      echo "── тесты SiMuse (отдельный воркспейс) ──"
      (cd crates/SiMuse && cargo test --workspace --release)
    else
      echo
      echo "тесты SiMuse пропущены (свой воркспейс, минуты) — scripts/verify.sh all"
    fi
    ;;
  *)
    echo "неизвестный режим \`$MODE\`; есть quick|all|fmt|lint" >&2
    exit 2
    ;;
esac

echo
echo "PASS"
