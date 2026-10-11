#!/bin/sh
# Generate a self-signed TLS certificate for the SimLab reverse proxy.
#
# Runs as a one-shot init container before the proxy starts (the `proxy-certs`
# service in docker-compose.yaml) and is a no-op once a certificate exists.
# To install a real certificate, drop the PEM files in nginx/certs/ as
# simlab.crt / simlab.key - this script then leaves them alone.
#
# It also runs directly on any host with openssl:
#
#   SIMLAB_TLS_CN=simlab.example.org \
#   SIMLAB_TLS_SAN=DNS:simlab.example.org,IP:10.0.0.5 \
#   sh nginx/scripts/generate-self-signed.sh nginx/certs

set -eu

CERT_DIR="${1:-${SIMLAB_CERT_DIR:-/certs}}"
CN="${SIMLAB_TLS_CN:-localhost}"
SAN="${SIMLAB_TLS_SAN:-DNS:localhost,IP:127.0.0.1}"
DAYS="${SIMLAB_TLS_DAYS:-825}"

CRT="$CERT_DIR/simlab.crt"
KEY="$CERT_DIR/simlab.key"

log() { echo "simlab-certs: $*"; }

if [ -s "$CRT" ] && [ -s "$KEY" ]; then
    log "$CRT already present - keeping it"
    exit 0
fi

mkdir -p "$CERT_DIR"

log "generating self-signed certificate (CN=$CN, SAN=$SAN, ${DAYS}d)"

# 825 days is the maximum lifetime Apple/Chrome accept for a server cert.
openssl req -x509 -newkey rsa:2048 -sha256 -days "$DAYS" -nodes \
    -keyout "$KEY" -out "$CRT" \
    -subj "/C=BR/O=SimLab/CN=$CN" \
    -addext "subjectAltName=$SAN" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth"

chmod 644 "$CRT"
chmod 600 "$KEY"

log "wrote $CRT and $KEY"
