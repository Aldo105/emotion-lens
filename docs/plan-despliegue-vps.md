# Plan de despliegue en VPS

Estado: pendiente de contratar el servidor. Fecha: 2026-09-29.

Objetivo: tener EmotionLens en una URL pública con HTTPS y que cada push a
`master` se despliegue solo mediante `.github/workflows/deploy.yml`.

Mientras no se complete este plan, ese workflow **falla en cada push**: no es un
error del código, sino que faltan los secretos `VPS_HOST`, `VPS_USER` y
`VPS_SSH_KEY`. GitHub manda un correo "Deploy to VPS: All jobs have failed"
cada vez.

---

## 1. Elegir servidor

Requisito mínimo: **2 vCPU / 4 GB de RAM**. PyTorch en CPU y MediaPipe corren en
el servidor, sin GPU.

Precios consultados el 2026-09-29, sin impuestos:

| Proveedor | Plan | Ubicación | USD/mes |
|---|---|---|---|
| Hetzner | CX23 (2 vCPU, 4 GB) | Alemania / Finlandia | ~6.49 |
| **Hetzner** | **CX33 (4 vCPU, 8 GB)** | Alemania / Finlandia | **~9.99** |
| Hetzner | CPX21 (3 vCPU, 4 GB) | EE. UU. | ~37.49 |
| DigitalOcean | Basic (2 vCPU, 4 GB) | EE. UU. / Canadá | 24.00 |

- Los planes baratos de Hetzner (serie CX) **no existen en sus centros de EE. UU.**
  La IPv4 se cobra aparte.
- Recomendado: **Hetzner CX33**. Tiene margen para varias sesiones a la vez por
  menos de la mitad que DigitalOcean. Su coste es la latencia: el servidor queda
  en Europa y el análisis en vivo envía frames de la webcam de forma continua.
- Si la latencia desde México resulta molesta: **DigitalOcean, $24/mes, en EE. UU.**
- Imagen: **Ubuntu 22.04**. `setup_vps.sh` instala `python3.11` con apt, y
  en 24.04 ese paquete no está en los repositorios oficiales.

Fuentes: [Hetzner, ajuste de precios de junio 2026](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/),
[DigitalOcean, precios de droplets](https://www.digitalocean.com/pricing/droplets).

## 2. Dominio y HTTPS

HTTPS es obligatorio: el navegador bloquea la cámara (`getUserMedia`) en
orígenes sin TLS.

- Opción de pago: dominio `.com`, ~$10–15 USD/año, con un registro A apuntando
  a la IP del VPS.
- Opción gratuita: subdominio de [DuckDNS](https://www.duckdns.org). Funciona
  con el certificado gratuito de Let's Encrypt que pide `setup_vps.sh`.

## 3. Arreglos en `deploy/` antes de instalar

Detectados al revisar los scripts. Hay que hacerlos **antes** del paso 4:

- [x] **Permiso para reiniciar el servicio.** `deploy.sh` ejecuta
  `sudo systemctl restart emotionlens` como el usuario `emotionlens`, pero
  `setup_vps.sh` no le da ese permiso. Desde GitHub Actions, sin contraseña,
  el despliegue fallaría en ese paso. Añadir a `setup_vps.sh` una regla de
  sudoers limitada a ese comando:
  ```
  emotionlens ALL=(root) NOPASSWD: /usr/bin/systemctl restart emotionlens, /usr/bin/systemctl status emotionlens *
  ```
  en `/etc/sudoers.d/emotionlens`, con permisos 0440 y validada con `visudo -cf`.
  Hecho: `setup_vps.sh` crea la regla y la valida con `visudo`.
- [x] **Email de Certbot.** Hecho: `setup_vps.sh` lo recibe como tercer argumento.
- [x] **Login.** La app exige usuario y contraseña (`docs/superpowers/specs/2026-09-29-login-design.md`). `setup_vps.sh` los pide durante la instalación.
- [ ] **El modelo no viaja con git.** `backend/ml/saved_models/emotion_cnn.pt`
  está en `.gitignore`, así que el VPS no lo tendría al clonar. Sin él, el
  clasificador cae en silencio al modo heurístico (mucho peor). `face_landmarker.task`
  sí está versionado. Hay que copiarlo a mano tras la instalación (paso 5) y
  cada vez que se promueva un modelo nuevo. Si esto se vuelve frecuente, valorar
  publicarlo como adjunto de una *release* de GitHub y que `deploy.sh` lo descargue.

## 4. Instalar en el VPS (una sola vez)

```bash
ssh root@<IP>
git clone https://github.com/Aldo105/emotion-lens.git /tmp/emotion-lens
sudo bash /tmp/emotion-lens/deploy/setup_vps.sh <dominio> https://github.com/Aldo105/emotion-lens.git <tu-email>
```

Instala dependencias CPU, PostgreSQL, el servicio systemd con `Restart=always`,
Nginx, Certbot y el firewall. **Guardar la contraseña de la base de datos** que
imprime al final. Al final pide el **usuario y la contraseña del login**: la contraseña no se muestra en pantalla y se guarda solo como hash.

## 5. Copiar el modelo

Desde la máquina local (PowerShell o Git Bash):

```bash
scp backend/ml/saved_models/emotion_cnn.pt root@<IP>:/home/emotionlens/emotion-lens/backend/ml/saved_models/
ssh root@<IP> "chown emotionlens:emotionlens /home/emotionlens/emotion-lens/backend/ml/saved_models/emotion_cnn.pt && systemctl restart emotionlens"
```

Comprobar en `journalctl -u emotionlens -n 50` que aparece
`[OK] Emotion CNN loaded from ...emotion_cnn.pt`.

## 6. Verificar

- [ ] Sin sesión, `https://<dominio>/` lleva al login y `https://<dominio>/api/sessions/` responde 401.
- [ ] Con el usuario y la contraseña, se entra y el botón de cerrar sesión aparece en la cabecera.
- [ ] `https://<dominio>` carga el dashboard y pide permiso de cámara.
- [ ] `https://<dominio>/api/health` responde.
- [ ] Una sesión en vivo calibra y muestra emociones. Tras calibrar aparece
      `[OK] CNN neutral profile set from N frames` en los logs.
- [ ] Probar en reposo tras calibrar: el punto débil medido del modelo actual son
      las caras neutras (65% en CREMA-D, ver `backend/app/references.py`).

## 7. Despliegue automático desde GitHub

1. Generar en local una clave **solo para desplegar**, sin contraseña:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/emotionlens_deploy -N "" -C "github-actions-deploy"
   ```
2. Autorizarla para el usuario `emotionlens` en el VPS:
   ```bash
   ssh root@<IP> "mkdir -p /home/emotionlens/.ssh && cat >> /home/emotionlens/.ssh/authorized_keys && chown -R emotionlens:emotionlens /home/emotionlens/.ssh && chmod 700 /home/emotionlens/.ssh && chmod 600 /home/emotionlens/.ssh/authorized_keys" < ~/.ssh/emotionlens_deploy.pub
   ```
3. En GitHub → repositorio → Settings → Secrets and variables → Actions,
   crear tres secretos:
   - `VPS_HOST`: IP o dominio del VPS.
   - `VPS_USER`: `emotionlens`.
   - `VPS_SSH_KEY`: el contenido **completo** de `~/.ssh/emotionlens_deploy`
     (la clave privada, sin `.pub`).
4. Hacer un push (o relanzar la última ejecución fallida desde la pestaña
   Actions) y comprobar que el job `deploy` sale en verde.

La clave privada solo vive en tu máquina y en el secreto de GitHub. No se sube
al repositorio.

## 8. Después

- Actualizar a mano: `sudo -u emotionlens bash deploy/deploy.sh`.
- Logs: `journalctl -u emotionlens -f`. Estado: `systemctl status emotionlens`.
- Nuevo modelo: repetir el paso 5. El despliegue automático no lo copia.
- Cambiar la contraseña: `sudo -u emotionlens ./venv/bin/python -m backend.app.auth crear-credenciales --env .env` en `/home/emotionlens/emotion-lens` y `sudo systemctl restart emotionlens`. Cierra todas las sesiones abiertas.
- Coste recurrente: servidor ~$10/mes (Hetzner CX33) o $24/mes (DigitalOcean),
  más el dominio si es de pago.
