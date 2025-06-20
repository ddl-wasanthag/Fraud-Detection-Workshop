#!/usr/bin/env bash
set -euo pipefail

# to use, run PORT=8501 bash app.sh
# run this if needed pkill -f streamlit

# Default to prod port 8888, but allow override via ENV or CLI arg
PORT="${PORT:-${1:-8888}}"

mkdir -p .streamlit

cat > .streamlit/config.toml <<EOF
[browser]
gatherUsageStats = true

[server]
address = "0.0.0.0"
port = $PORT
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#543FDD"              # purple5000
backgroundColor = "#FFFFFF"           # neutralLight50
secondaryBackgroundColor = "#FAFAFA"  # neutralLight100
textColor = "#2E2E38"                 # neutralDark700
EOF

cat > .streamlit/pages.toml <<EOF
[[pages]]
path = "fraud_detection.py"
name = "Fraud Detection"

EOF

streamlit run apps/dashboard.py