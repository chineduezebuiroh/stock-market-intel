    #!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  source .venv/Scripts/activate
fi
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Environment ready. Activate it with: source .venv/bin/activate"
