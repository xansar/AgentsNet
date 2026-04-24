#!/usr/bin/env bash
set -euo pipefail

WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-https://www.feishu.cn/flow/api/trigger-webhook/b0d9ebf1d95a32009be1b67d0d6ef493}"
TITLE=""
MESSAGE=""
FIELDS=()

usage() {
    cat <<'USAGE'
Usage:
  scripts/feishu_notify.sh --title "Title" --message "Message" [--field key=value ...]

Environment:
  FEISHU_WEBHOOK_URL      Override the default Feishu webhook URL.
  FEISHU_NOTIFY_DRY_RUN   Set to 1 to print the JSON payload without calling curl.

Notes:
  Flow trigger webhook URLs receive a flat JSON object with title, message,
  text, and any repeated --field key=value pairs.
  Custom bot webhook URLs receive the Feishu text-message payload format.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --title)
            TITLE="${2:-}"
            shift 2
            ;;
        --message)
            MESSAGE="${2:-}"
            shift 2
            ;;
        --field)
            if [[ "${2:-}" != *=* || "${2:-}" == "="* ]]; then
                echo "Invalid --field value. Expected key=value." >&2
                usage >&2
                exit 2
            fi
            FIELDS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$MESSAGE" ]]; then
    echo "Missing required --message" >&2
    usage >&2
    exit 2
fi

if [[ -z "$WEBHOOK_URL" ]]; then
    echo "FEISHU_WEBHOOK_URL is not set; skip Feishu notification."
    exit 0
fi

if [[ -n "$TITLE" ]]; then
    TEXT="${TITLE}

${MESSAGE}"
else
    TEXT="$MESSAGE"
fi

if [[ "$WEBHOOK_URL" == *"/flow/api/trigger-webhook/"* ]]; then
    PAYLOAD="$(python -c '
import json, sys

title, message, text, *fields = sys.argv[1:]
payload = {"title": title, "message": message, "text": text}
for field in fields:
    key, value = field.split("=", 1)
    payload[key] = value
print(json.dumps(payload, ensure_ascii=False))
' "$TITLE" "$MESSAGE" "$TEXT" "${FIELDS[@]}")"
else
    PAYLOAD="$(python -c '
import json, sys

print(json.dumps({"msg_type": "text", "content": {"text": sys.argv[1]}}, ensure_ascii=False))
' "$TEXT")"
fi

if [[ "${FEISHU_NOTIFY_DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "$PAYLOAD"
    exit 0
fi

curl -fsS \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$WEBHOOK_URL" \
    >/dev/null
