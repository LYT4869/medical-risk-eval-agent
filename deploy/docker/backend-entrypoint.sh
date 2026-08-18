#!/usr/bin/env bash
set -euo pipefail

for attempt in $(seq 1 60); do
  if MYSQL_PWD="${TREESEM_DB_PASSWORD}" mysqladmin \
      --host="${TREESEM_DB_HOST}" --port="${TREESEM_DB_PORT}" \
      --user="${TREESEM_DB_USER}" --protocol=TCP ping --silent; then
    break
  fi
  if [[ "${attempt}" == "60" ]]; then
    echo "treeSem database did not become ready" >&2
    exit 1
  fi
  sleep 1
done

/app/scripts/migrate_treesem_db.sh
exec "$@"
