#!/bin/sh
set -e

RAW_CSV=/data/raw/accepted_2007_to_2018q4.csv
PROCESSED_CHECK=/data/processed/2007.csv

if [ ! -f "$RAW_CSV" ]; then
    mkdir -p /data/raw
    kaggle datasets download -d wordsforthewise/lending-club -p /data/raw/ --unzip
    # Le zip peut extraire dans un sous-dossier — on cherche le CSV et on le remonte
    ACTUAL=$(find /data/raw -iname "accepted_2007_to_2018q4.csv" ! -type d | head -1)
    if [ -z "$ACTUAL" ]; then
        echo "[ERROR] CSV introuvable après extraction :" >&2
        find /data/raw -maxdepth 3 >&2
        exit 1
    fi
    [ "$ACTUAL" != "$RAW_CSV" ] && mv "$ACTUAL" "$RAW_CSV"
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
