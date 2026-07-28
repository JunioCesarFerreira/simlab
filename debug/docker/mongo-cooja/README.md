# MongoDB, Cooja and Monitoring Containers

Use this docker-compose to generate a minimal infrastructure with MongoDB (locally connectable), six Cooja containers, and the monitoring pair (cAdvisor + Prometheus).

This way, you can run Python components directly on your localhost for debugging and new features.

Prometheus is published on `localhost:9090`, which is the default URL the runtime-telemetry collector uses when `IS_DOCKER` is not set (override with `PROMETHEUS_URL` if needed). If Prometheus is not running, telemetry is skipped with a single warning — experiments are not affected.

The `simlab.group=simulation` labels on the Cooja services are required for telemetry: the collector filters cAdvisor series by `container_label_simlab_group=~"simulation|backend"`.
