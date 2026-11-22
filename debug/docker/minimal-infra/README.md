# Minimal Docker Compose for local use

A minimal structure for testing on localhost: MongoDB, REST API, mo-engine,
master-node and three Cooja containers. There is no GUI, proxy or monitoring,
and the code is bind-mounted from the checkout.

```bash
cd debug/docker/minimal-infra
docker compose up -d --build
```

The API is on `http://localhost:8000` (Swagger in `/docs`, key
`simlab-api-key42`).

What differs from the [main stack](../../../README.md):

- **Every port is published on all interfaces**, MongoDB without
  authentication included. Use it only on a trusted machine.
- **No Docker socket**, so no container discovery: the master-node probes the
  three workers it registers itself from `NUMBER_OF_CONTAINERS`. Change the
  services and that variable together. Without discovery, a worker that ends a
  run only returns to the pool at the next health cycle.
- **The master-node has no restart policy.** If you register hosts by name,
  set `DNS_SELF_RESTART=false`: the
  [self-restart](../../../master-node/README.md#the-master-node-cannot-resolve-host-names)
  would leave it stopped.
- The variables not set in the file take the code's defaults, which are not the
  main stack's (placement policy and JVM heap, for example; see the
  [master-node configuration](../../../master-node/README.md#configuration-environment-variables)).
- It reuses the main stack's container and network names: do not run both.
