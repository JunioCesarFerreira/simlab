# MongoDB, Cooja and Monitoring Containers

Use this docker-compose to generate a minimal infrastructure with MongoDB (locally connectable), eight Cooja containers, and the monitoring pair (cAdvisor + Prometheus).

This way, you can run Python components directly on your localhost for debugging and new features. Its ports are published on all interfaces, and its container names are the main stack's: do not run both.

The replica set advertises `localhost:27017`, so the components' default `MONGO_URI` works from the host here. `INFRA_KEY_PATH` has to point to a path the master-node can write on the host.

Prometheus is published on `localhost:9090`, which is the default URL the runtime-telemetry collector uses when `IS_DOCKER` is not set (override with `PROMETHEUS_URL` if needed). If Prometheus is not running, the telemetry is recorded as `unavailable` with a single warning — experiments are not affected.

cAdvisor reads containerd here, and those series carry no Docker labels, so the collector's default filter (`container_label_simlab_group=~"simulation|backend"`) matches nothing. Start the mo-engine on the host with:

```bash
export TELEMETRY_CONTAINER_FILTER='image=~"docker.io/juniocesarferreira/simulation-cooja.*"'
```

The telemetry then covers the Cooja containers only. See [RUNTIME_METRICS.md](../../../docs/markdown/RUNTIME_METRICS.md).
