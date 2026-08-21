#!/usr/bin/env bash
# Phase 4E — generate a self-signed TLS cert for local HTTPS testing.
# Produces deploy/nginx/certs/{server.crt,server.key}. NOT for production
# (browsers will warn on the self-signed cert) — mount real certs in prod.
set -euo pipefail

CERT_DIR="$(cd "$(dirname "$0")" && pwd)/certs"
mkdir -p "$CERT_DIR"

if [[ -f "$CERT_DIR/server.crt" && -f "$CERT_DIR/server.key" ]]; then
  echo "certs already exist in $CERT_DIR (delete them to regenerate)"
  exit 0
fi

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$CERT_DIR/server.key" \
  -out    "$CERT_DIR/server.crt" \
  -days 365 \
  -subj "/C=US/ST=Dev/L=Dev/O=AutoML-QO/CN=localhost"

echo "wrote $CERT_DIR/server.crt and server.key"
