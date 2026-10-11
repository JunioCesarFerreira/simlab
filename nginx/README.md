# SimLab reverse proxy

nginx terminates TLS in front of the whole stack. It is the **only** ingress:
the SPA, the REST API (Swagger included) and Grafana no longer publish host
ports, so every request from outside the machine arrives here over HTTPS.

## Routes

| URL | Upstream | Notes |
| --- | --- | --- |
| `https://<host>/` | `gui:80` | Vue SPA; deep links resolved by the gui container's own `try_files` |
| `https://<host>/api/v1/...` | `restapi:8000` | REST API |
| `https://<host>/docs` | `restapi:8000` | Swagger UI (`/docs/oauth2-redirect` included) |
| `https://<host>/redoc` | `restapi:8000` | ReDoc |
| `https://<host>/openapi.json` | `restapi:8000` | OpenAPI schema |
| `https://<host>/grafana/` | `grafana:3000` | Dashboards, served from a sub-path |
| `https://<host>/healthz` | — | Answered by the proxy itself |
| `http://<host>/*` | — | `301` to `$SIMLAB_PUBLIC_URL` |

`http://<host>/healthz` stays on plain HTTP so the compose healthcheck can
probe the proxy without dealing with the certificate.

## Layout

```
nginx/
├── nginx.conf                          http-level config: gzip, logging,
│                                       client_max_body_size, websocket map
├── conf.d/simlab.conf                  the :443 server block and route map
├── templates/
│   └── 00-http-redirect.conf.template  :80 redirect, rendered at container
│                                       start so it can carry a custom port
├── snippets/
│   ├── tls.conf                        certificate paths + cipher policy
│   ├── security-headers.conf           HSTS, nosniff, frame options, referrer
│   └── proxy-common.conf               shared proxy_set_header + timeouts
├── scripts/generate-self-signed.sh     one-shot certificate bootstrap
└── certs/                              simlab.crt / simlab.key (git-ignored)
```

## Certificates

On first `docker compose up`, the `proxy-certs` init container writes a
self-signed pair into `certs/` and exits. It **skips itself whenever
`certs/simlab.crt` already exists**, so installing a real certificate is just:

```bash
cp fullchain.pem nginx/certs/simlab.crt
cp privkey.pem   nginx/certs/simlab.key
docker compose restart proxy
```

Regenerate the self-signed pair for a different hostname:

```bash
rm nginx/certs/simlab.crt nginx/certs/simlab.key
# values come from .env - see SIMLAB_TLS_CN / SIMLAB_TLS_SAN
docker compose up proxy-certs
docker compose restart proxy
```

The script also runs directly on any host with `openssl`:

```bash
SIMLAB_TLS_CN=simlab.example.org \
SIMLAB_TLS_SAN=DNS:simlab.example.org,IP:10.0.0.5 \
sh nginx/scripts/generate-self-signed.sh nginx/certs
```

The private key is written by a root-owned container, so it lands as
`root:root 0600` on the host. Removing it needs `sudo` on some setups.

## Configuration

All of it lives in `.env` at the repository root (see `.env.example`); every
value has a working default, so the stack comes up without one.

`SIMLAB_PUBLIC_URL` is the one that matters. It feeds both the HTTP→HTTPS
redirect and Grafana's `root_url`, and **must include the port** whenever HTTPS
is published somewhere other than 443:

```ini
SIMLAB_HTTPS_PORT=8443
SIMLAB_PUBLIC_URL=https://simlab.example.org:8443
```

Get this wrong and Grafana still loads, but its own redirects (login, asset
URLs) point off the proxy.

## Reaching the API from scripts

`util/`, `firmware/` and `pareto-analysis/` default to
`https://localhost/api/v1` and verify TLS against `nginx/certs/simlab.crt`,
which is why the self-signed certificate does not force `verify=False`
anywhere. Override with:

| Variable | Effect |
| --- | --- |
| `SIMLAB_API_BASE` | API base URL (default `https://localhost/api/v1`) |
| `SIMLAB_CA_BUNDLE` | Trust a different CA bundle |
| `SIMLAB_TLS_VERIFY=false` | Skip verification entirely — last resort |

For `curl`, either pass `--cacert nginx/certs/simlab.crt` or accept the
self-signed certificate with `-k`.

## Design notes

**Runtime DNS resolution.** Upstreams are addressed through `$upstream_*`
variables with `resolver 127.0.0.11`. A literal hostname in `proxy_pass` is
resolved once at startup, so recreating a backend (`docker compose up -d
--build gui`) would leave the proxy 502-ing on a stale IP until it was
restarted. Verified: moving the gui container to a new IP recovers within the
10s TTL, no proxy restart.

**Relative redirects.** `absolute_redirect off` keeps nginx from rebuilding
`Location` headers out of the *container* port (443), which would break any
deployment publishing HTTPS elsewhere.

**Grafana sub-path.** Grafana runs with `GF_SERVER_SERVE_FROM_SUB_PATH=true`
and therefore expects the `/grafana` prefix intact — `proxy_pass` must not
rewrite the URI. This also moves its health endpoint to
`/grafana/api/health`, which is what the compose healthcheck probes. Grafana's
own `X-Frame-Options: deny` is dropped with `proxy_hide_header` so responses do
not carry two conflicting values.

**Upload size.** `client_max_body_size 100m` at the http level; the 1m default
rejects firmware/source uploads. Note the gui container's internal nginx still
has the 1m default, but uploads never traverse it — `/api/` is routed straight
to `restapi`.

**No Content-Security-Policy.** The Vue build and Grafana both need inline
styles/eval, and a wrong policy fails silently in the browser. Add one
per-route only after checking the console.

**IPv4 only.** `listen 443 ssl` without an `[::]` counterpart, matching
`gui/simlab/nginx.conf`: the compose networks have no IPv6 and binding an
unavailable address family aborts startup.

## Remaining exposure

Everything else is bound to `127.0.0.1` rather than removed, so local debugging
still works while nothing is reachable off-box:

| Service | Bind | Reach it remotely with |
| --- | --- | --- |
| Prometheus | `127.0.0.1:9090` | `ssh -L 9090:localhost:9090 <vm>` |
| cAdvisor | `127.0.0.1:8080` | `ssh -L 8080:localhost:8080 <vm>` |
| MongoDB | `127.0.0.1:27017` | `ssh -L 27017:localhost:27017 <vm>` |
| mo-engine / master-node SSH | `127.0.0.1:2219`, `:2220` | `ssh -J <vm> ...` |
| cooja1–16 SSH | `127.0.0.1:2231`–`:2246` | `ssh -J <vm> ...` |

Known gaps that this proxy does **not** close:

- **MongoDB runs without authentication.** Loopback binding is a mitigation,
  not a fix; anyone with a shell on the host has full database access.
- **The API key is not a secret.** `SIMLAB_API_KEY` is baked into the SPA
  bundle at build time and therefore readable by anyone who can load the page.
- **CORS is still `allow_origins=["*"]`** in `rest-api/main.py`. Now that the
  SPA and the API share an origin behind this proxy, that can be narrowed to
  `SIMLAB_PUBLIC_URL` — left alone here because it would break the
  `npm run dev` workflow, which serves the SPA from a different origin.
