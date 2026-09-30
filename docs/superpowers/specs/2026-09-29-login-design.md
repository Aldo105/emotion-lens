# Login de un solo usuario — diseño

Fecha: 2026-09-29. Estado: aprobado en conversación, pendiente de plan.

## Por qué

EmotionLens va a publicarse en un VPS (`docs/plan-despliegue-vps.md`) y hoy no tiene
ninguna autenticación. Con la URL pública, cualquiera podría:

- listar todas las sesiones (`GET /api/sessions/`);
- descargar reportes PDF y CSV, notas del entrevistador y vídeos con la cara de los
  candidatos (`/api/reports/{id}/evm-video`, `/micro-highlights-video`);
- borrar sesiones;
- saturar la CPU subiendo vídeos de hasta 500 MB.

Los identificadores son enteros consecutivos, así que recorrerlos es trivial. Son datos
biométricos de un proceso de contratación: sensibles según la LFPDPPP.

## Decisiones tomadas con el usuario

| Tema | Decisión |
|---|---|
| Usuarios | **Uno solo** (habrá un único entrevistador). Sin tabla de usuarios ni roles. |
| Credenciales | **Usuario y contraseña**. |
| Duración de la sesión | **12 horas**, sin casilla "recordarme". |
| Local | **Sin login**: `AUTH_ENABLED=false` en el `.env` local. Por defecto está activado. |
| Enfoque | **Middleware en la app con cookie firmada** (se descartan sesiones en BD y `auth_request` de Nginx). HTTP Basic de Nginx se descarta porque su diálogo nativo no se puede diseñar. |
| Diseño visual | **Opción B, dividida**: franja ilustrada a la izquierda y formulario a la derecha. |
| Alcance extra | Incluir en `setup_vps.sh` la regla de sudoers para `systemctl restart emotionlens` y el email de Certbot como argumento. |

## Arquitectura

### `backend/app/auth.py` (nuevo)

Unidad sin dependencias de FastAPI, testeable por separado:

- `hash_password(password) -> str` / `verify_password(password, stored) -> bool`:
  `hashlib.scrypt` de la biblioteca estándar (n=2**14, r=8, p=1, sal de 16 bytes).
  Formato almacenado: `scrypt:<n>:<r>:<p>:<sal_b64>:<hash_b64>` (con `:` y no `$`, para que
  el lector del `.env` no interprete nada). Comparación con `hmac.compare_digest`.
- `make_token(...) -> str` / `token_is_valid(token, ...) -> bool`: el token
  es `base64url(username|expira_epoch)` + `.` + firma HMAC-SHA256. La clave de firma es
  `AUTH_SECRET_KEY + AUTH_PASSWORD_HASH`, así que cambiar cualquiera de las dos invalida
  todas las sesiones. `token_is_valid` devuelve falso si la firma no cuadra, el formato es
  inválido, el usuario no coincide o el token caducó.
- `LoginRateLimiter`: en memoria, por IP. **5 fallos en 15 minutos → bloqueo de 15
  minutos** para esa IP. Un login correcto borra el contador de esa IP. Reloj inyectable
  para los tests.
- CLI: `python -m backend.app.auth crear-credenciales` pide el usuario, y la contraseña
  dos veces con `getpass`, e imprime las cuatro líneas para el `.env`:
  `AUTH_ENABLED=true`, `AUTH_USERNAME`, `AUTH_PASSWORD_HASH` y `AUTH_SECRET_KEY`
  (esta última con `secrets.token_urlsafe(48)`). Rechaza contraseñas de menos de 8
  caracteres y usuarios vacíos o con espacios. Opción `--env RUTA` para escribirlas
  directamente en ese archivo, sustituyendo las existentes. Esa opción la usa
  `setup_vps.sh`.

### Configuración (`backend/app/config.py`)

Nuevos campos de `Settings`:

- `auth_enabled: bool = True`
- `auth_username: str = ""`
- `auth_password_hash: str = ""`
- `auth_secret_key: str = ""`
- `auth_session_hours: int = 12`

**Validación en el arranque de la app**, en el `lifespan`, no al importar: si
`auth_enabled` es verdadero y falta cualquiera de los tres valores, la app no arranca y
el mensaje de error indica el comando `crear-credenciales`. Así nunca queda abierta por
un olvido.

### Middleware (`backend/app/auth_middleware.py`, nuevo)

Middleware ASGI puro, porque debe cubrir también el ámbito `websocket`, que
`BaseHTTPMiddleware` no intercepta. Se registra en `main.py`.

- Con `auth_enabled` en falso, deja pasar todo sin tocarlo.
- **Rutas públicas**, las únicas accesibles sin sesión:
  - `GET /login.html`
  - `/css/styles.css`
  - `/js/login.js`, `/js/i18n.js` y `/js/icons.js`
  - `/i18n/*.json`
  - `POST /api/auth/login`
  - `GET /api/auth/status`
  - `GET /api/health`
- **El resto** exige una cookie `el_session` válida:
  - página HTML sin sesión → `302` a `/login.html?next=<ruta original>`;
  - `/api/*` sin sesión → `401 {"detail": "not_authenticated"}`;
  - WebSocket sin sesión → se acepta y se cierra en el acto con código `4401`, sin
    procesar nada. (Cerrar antes de aceptar llega al navegador como un 1006 genérico y el
    frontend no podría distinguirlo de una caída de red.)
- `next` solo acepta rutas relativas que empiecen por `/` y no por `//`, para evitar que
  el login redirija a otros dominios.

### Endpoints (`backend/app/routers/auth.py`, nuevo, prefijo `/api/auth`)

- `POST /login` con `{username, password}`:
  - IP bloqueada → `429 {"detail": "too_many_attempts", "retry_after": segundos}`;
  - datos incorrectos → `401 {"detail": "invalid_credentials"}` (el mismo mensaje sea
    cual sea el campo que falló) y se registra el fallo;
  - datos correctos → `204` con `Set-Cookie: el_session=<token>; HttpOnly;
    SameSite=Strict; Path=/; Max-Age=43200`, y `Secure` cuando la conexión es HTTPS.
- `POST /logout` → `204` y borra la cookie.
- `GET /status` (pública) → `{"enabled": bool}`. Solo revela si el login está activado,
  que no es un dato sensible.

`Max-Age` se deriva de `auth_session_hours` (12 h = 43200 s).

**Proxy.** La IP del cliente y si la conexión es HTTPS se leen de `X-Real-IP` y
`X-Forwarded-Proto` **solo si la conexión viene de 127.0.0.1**, es decir, de Nginx. Si
no, se usan la IP del socket y el esquema real.

## Frontend

- `frontend/login.html` y `frontend/js/login.js` (nuevos). Los estilos del login se
  añaden al final de `frontend/css/styles.css` y reutilizan sus variables.
- **Diseño B:**
  - Izquierda: degradado con `--accent-rgb`, la línea del logo (el mismo trazo que el
    icono `activity`), la frase "Lectura de expresiones, con sus límites a la vista" y la
    nota "Herramienta de investigación · no es un dispositivo médico".
  - Derecha: logo "EmotionLens", los campos Usuario y Contraseña (esta con botón de
    mostrar/ocultar), el botón "Entrar" y una línea para errores.
  - Por debajo de 720 px de ancho, la franja pasa arriba, con menos altura, y el
    formulario ocupa todo el ancho.
- **Idioma y tema:** conmutadores ES/EN y claro/oscuro arriba a la derecha. Usan las
  mismas claves de `localStorage` que la app (`emotionlens-lang` y `emotionlens-theme`).
  Los textos nuevos van bajo `login.*` en `i18n/es.json` y `en.json`.
- **Comportamiento del formulario:**
  - el foco empieza en Usuario y Enter envía;
  - `autocomplete="username"` y `"current-password"`;
  - mientras se envía, el botón se desactiva y dice "Entrando…";
  - si falla, "Usuario o contraseña incorrectos" y se vacía la contraseña;
  - con un `429`, "Demasiados intentos. Vuelve a intentarlo en 15 minutos.";
  - si sale bien, redirige a `next` (validado igual que en el servidor) o a `/`.
- **En la app:**
  - un botón "Cerrar sesión" en la cabecera, junto a los de idioma y tema. Solo aparece
    si `GET /api/auth/status` devuelve `enabled: true`;
  - un envoltorio global de `fetch` en `app.js`: ante cualquier `401` redirige a
    `/login.html?next=<ruta actual>`;
  - `websocket.js`: si el socket se cierra con `4401`, hace la misma redirección.

## Despliegue

- **`.env.example`:** nueva sección `# ── Auth ──` con las cuatro variables comentadas y
  la indicación de usar `crear-credenciales`.
- **`deploy/setup_vps.sh`:**
  - tercer argumento obligatorio, el email para Certbot: `setup_vps.sh <dominio>
    <repo_url> <email>`;
  - tras escribir el `.env`, si todavía no tiene `AUTH_PASSWORD_HASH`, ejecuta
    `crear-credenciales --env .env` de forma interactiva, antes de arrancar el servicio;
  - crea `/etc/sudoers.d/emotionlens`, con permisos 0440 y validado con `visudo -cf`,
    que permite a `emotionlens` ejecutar sin contraseña solo
    `systemctl restart emotionlens` y `systemctl status emotionlens *`.
- **`deploy/nginx.conf.template`:** añadir `X-Real-IP` y `X-Forwarded-Proto` también en
  la ubicación `/ws/`, por coherencia (la cookie ya llega en ambas).
- **`.env` local:** se crea con `AUTH_ENABLED=false`. No se versiona.
- **`docs/plan-despliegue-vps.md`:** el login pasa a formar parte de la instalación
  (paso 4), los arreglos del paso 3 se marcan como hechos y se añade la comprobación
  "sin sesión, `/api/sessions/` responde 401".

## Errores y casos límite

- Cookie manipulada, caducada o de un usuario anterior → se trata como si no hubiera
  sesión.
- Si el servidor se reinicia, el limitador de intentos, que vive en memoria, se vacía. Se
  acepta: un solo usuario y un solo proceso (gunicorn con 1 worker).
- Si la sesión caduca durante una grabación en vivo, el WebSocket ya abierto sigue hasta
  cerrarse, porque la comprobación es al conectar. Las llamadas a la API que vengan
  después redirigen al login.
- `evaluate.py` y demás scripts locales no se ven afectados, porque en local el login
  está desactivado.

## Pruebas

- `auth.py`:
  - hash y verificación, correcta e incorrecta;
  - token válido, caducado, con la firma alterada, con otro usuario y tras cambiar la
    contraseña;
  - el limitador: bloquea al quinto fallo, desbloquea a los 15 minutos, un éxito
    reinicia el contador, e IPs independientes.
- **Middleware y endpoints**, con `TestClient` y configuración fijada explícitamente en
  el test (sin depender de `.env`):
  - las rutas públicas pasan;
  - `/api/sessions/` responde 401 sin cookie y 200 con cookie;
  - una página HTML responde 302 con `next`;
  - un `next` externo se rechaza;
  - el WebSocket se rechaza sin cookie;
  - el login correcto pone la cookie con los atributos correctos, y `Secure` solo detrás
    de HTTPS;
  - el sexto intento responde 429;
  - el logout borra la cookie;
  - con `auth_enabled=false` todo pasa;
  - con el login activado y sin credenciales, la app no arranca.
- **Cabeceras de proxy:** solo se creen si la conexión viene de 127.0.0.1.
- **Frontend:** comprobación manual en el navegador con el tema claro y el oscuro, en ES
  y en EN, en móvil y en escritorio, con error de credenciales y con bloqueo.

## Fuera de alcance

- Varios usuarios, roles o dueños de sesiones.
- Recuperar la contraseña (se regenera con `crear-credenciales`).
- 2FA.
- Límite de subida de vídeos más allá del que impone el propio login.
