#!/bin/sh
set -e

RAW_CSV=/data/raw/accepted_2007_to_2018q4.csv
PROCESSED_CHECK=/data/processed/2007.csv

if [ ! -f "$RAW_CSV" ]; then
    mkdir -p /data/raw
    kaggle datasets download -d wordsforthewise/lending-club -p /data/raw/ --unzip

    # Kaggle peut extraire dans un dossier portant le même nom que le CSV
    if [ -d "$RAW_CSV" ]; then
        ACTUAL=$(find "$RAW_CSV" -iname "accepted_2007_to_2018q4.csv" ! -type d | head -1)
        [ -z "$ACTUAL" ] && ACTUAL=$(find "$RAW_CSV" -name "*.csv" ! -type d | head -1)
        cp "$ACTUAL" /tmp/lending_raw.csv
        rm -rf "$RAW_CSV"
        mv /tmp/lending_raw.csv "$RAW_CSV"
    fi

    if [ ! -f "$RAW_CSV" ]; then
        echo "[ERROR] CSV introuvable après extraction :" >&2
        find /data/raw -maxdepth 3 >&2
        exit 1
    fi
    echo "[OK] Données téléchargées : $RAW_CSV"
else
    echo "[OK] Raw data déjà présente, skip download."
fi

if [ ! -f "$PROCESSED_CHECK" ]; then
    echo ">> Split par année..."
    python - <<'EOF'
from pathlib import Path
from lending.data import load_raw, clean, split_by_year
df = load_raw(Path("/data/raw/accepted_2007_to_2018q4.csv"))
df = clean(df)
split_by_year(df, Path("/data/processed"))
print("[OK] Split terminé.")
EOF
else
    echo "[OK] Données processées déjà présentes, skip split."
fi
