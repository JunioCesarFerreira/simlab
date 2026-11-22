# Developer notes — legacy TSCH firmware

See [README.md](README.md) for what this firmware measures and why it is not
used.

## Build

The `Makefile` expects Contiki-NG two directories up (`CONTIKI = ../..`). Copy
the files to such a directory (in the `simulation-cooja` image, SimLab uses
`/opt/contiki-ng/tools/cooja`) and build each mote:

```sh
make root.cooja TARGET=cooja
make node.cooja TARGET=cooja
```

A plain `make` fails, because the default target names `udp-client` and
`udp-server`, which are not here. `TARGET=native` does not build either.

In Cooja, add one mote built from `root.c` and the others from `node.c`.

## Debug flags

Both are commented out in `node.c`; uncomment to use:

| Flag | Effect |
|---|---|
| `DEBUG_ENERGY_TIME_IN_SECONDS` | prints the Energest times in seconds |
| `DEBUG_RX_CALLBACK` | prints each packet's sender and size |

`DEBUG_PRINT_METRICS_PACKET` appears in the code but calls a function that does
not exist: do not define it.

## Post-processing

`COOJA.testlog` is not JSON Lines, so `pandas.read_json(lines=True)` fails on
it. The converter SimLab uses extracts the root's records (run from the
repository root, with `pandas` installed):

```python
from pathlib import Path
from pylib.cooja_builder.csv_converter import cooja_log_to_csv

df = cooja_log_to_csv(Path("COOJA.testlog"), Path("result.csv"))
print(df.groupby("node")[["total_mj", "latency_ms"]].mean())
```

## Adding a metric

1. Extend `node_metrics_packet_t` in `metrics-packet.h`.
2. Fill it in `fill_node_metrics_packet()` in `node.c`.
3. Print it in `metrics_print()` in `root.c`.

For new work, start from a v2 directory and follow the
[rules for changes](../README.md#rules-for-changes).

## References

- [Contiki-NG documentation](https://docs.contiki-ng.org/)
- RPL: RFC 6550. TSCH: IEEE 802.15.4 and RFC 7554.
