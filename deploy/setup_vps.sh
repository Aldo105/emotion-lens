#!/usr/bin/env bash
# EmotionLens — one-time provisioning for a fresh Ubuntu 22.04 CPU VPS.
#
# Run as root (or with sudo) on the VPS itself:
#   sudo bash deploy/setup_vps.sh your-domain.com https://github.com/your-org/emotion-lens.git
#
# What it does: installs system deps, creates the app user + venv, sets up
# PostgreSQL, installs the systemd service, configures Nginx, and requests
# a Let's Encrypt certificate. Re-run is safe (idempotent where practical).

set -euo pipefail

DOMAIN="${1:?Usage: setup_vps.sh <domain> <repo_url>}"
REPO_URL="${2:?Usage: setup_vps.sh <domain> <repo_url>}"
APP_USER="emotionlens"
APP_DIR="/home/${APP_USER}/emotion-lens"
DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -hex 16)}"

echo ">> Installing system packages..."
apt update && apt upgrade -y
apt install -y python3.11 python3.11-venv python3-pip git nginx \
    certbot python3-certbot-nginx libgl1 libglib2.0-0 \
    postgresql postgresql-contrib ufw

echo ">> Creating app user..."
id -u "$APP_USER" &>/dev/null || useradd -m -s /bin/bash "$APP_USER"

echo ">> Cloning repo and installing Python deps..."
sudo -u "$APP_USER" bash -c "
    set -e
    cd /home/$APP_USER
    if [ ! -d emotion-lens ]; then
        git clone '$REPO_URL' emotion-lens
    fi
    cd emotion-lens
    python3.11 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r deploy/requirements-cpu.txt
    mkdir -p data/uploads data/reports data/recordings
"

echo ">> Configuring PostgreSQL..."
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${APP_USER}'" | grep -q 1 \
    || sudo -u postgres psql -c "CREATE USER ${APP_USER} WITH PASSWORD '${DB_PASSWORD}';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${APP_USER}'" | grep -q 1 \
    || sudo -u postgres psql -c "CREATE DATABASE ${APP_USER} OWNER ${APP_USER};"

echo ">> Writing .env..."
if [ ! -f "$APP_DIR/.env" ]; then
    sed \
        -e "s#https://your-domain.com#https://${DOMAIN}#" \
        -e "s#postgresql+asyncpg://emotionlens:CHANGE_ME@localhost:5432/emotionlens#postgresql+asyncpg://${APP_USER}:${DB_PASSWORD}@localhost:5432/${APP_USER}#" \
        "$APP_DIR/.env.example" > "$APP_DIR/.env"
    chown "$APP_USER":"$APP_USER" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
fi

echo ">> Installing systemd service..."
mkdir -p /var/log/emotionlens
chown "$APP_USER":"$APP_USER" /var/log/emotionlens
sed \
    -e "s#APP_USER_PLACEHOLDER#${APP_USER}#g" \
    -e "s#APP_DIR_PLACEHOLDER#${APP_DIR}#g" \
    "$APP_DIR/deploy/emotionlens.service.template" > /etc/systemd/system/emotionlens.service
systemctl daemon-reload
systemctl enable emotionlens
systemctl restart emotionlens

echo ">> Configuring Nginx..."
sed "s/your-domain.com/${DOMAIN}/" "$APP_DIR/deploy/nginx.conf.template" \
    > /etc/nginx/sites-available/emotionlens
ln -sf /etc/nginx/sites-available/emotionlens /etc/nginx/sites-enabled/emotionlens
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo ">> Requesting HTTPS certificate..."
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "admin@${DOMAIN}" --redirect \
    || echo ">> Certbot failed automatically — run manually: certbot --nginx -d ${DOMAIN}"

echo ">> Configuring firewall..."
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

echo ""
echo "=================================================================="
echo " Done. Visit: https://${DOMAIN}"
echo " DB password (save this): ${DB_PASSWORD}"
echo " Service status: systemctl status emotionlens"
echo " Logs: journalctl -u emotionlens -f"
echo "=================================================================="
