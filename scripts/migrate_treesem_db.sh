#!/usr/bin/env bash
set -euo pipefail

: "${TREESEM_DB_HOST:=127.0.0.1}"
: "${TREESEM_DB_PORT:=3307}"
: "${TREESEM_DB_NAME:=treesem}"
: "${TREESEM_DB_USER:=treesem_app}"
: "${TREESEM_DB_PASSWORD:?TREESEM_DB_PASSWORD is required}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "${script_dir}/.." && pwd)"

for migration in "${repository_dir}"/db/migrations/*.sql; do
  version="$(basename "${migration}" .sql)"
  if [[ "${version}" != "001_m4_core" ]]; then
    applied="$(MYSQL_PWD="${TREESEM_DB_PASSWORD}" mysql \
      --host="${TREESEM_DB_HOST}" --port="${TREESEM_DB_PORT}" \
      --user="${TREESEM_DB_USER}" --database="${TREESEM_DB_NAME}" \
      --protocol=TCP --batch --skip-column-names \
      --execute="SELECT COUNT(*) FROM schema_migrations WHERE version='${version}'")"
    if [[ "${applied}" == "1" ]]; then
      continue
    fi
  fi
  MYSQL_PWD="${TREESEM_DB_PASSWORD}" mysql \
    --host="${TREESEM_DB_HOST}" \
    --port="${TREESEM_DB_PORT}" \
    --user="${TREESEM_DB_USER}" \
    --database="${TREESEM_DB_NAME}" \
    --protocol=TCP \
    < "${migration}"
done
