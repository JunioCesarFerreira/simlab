## Commands Examples

**Localhost**

```bash
py plot_pareto_3obj.py --expid 695d27d1f2ebb0848d08df3f --api-key api-password
```

```bash
py hv_gd_3obj.py --expid 695d27d1f2ebb0848d08df3f --api-key api-password --ref-point 9000000 1100 0
```

```bash
py plot_parallel_coords.py --expid 695d27d1f2ebb0848d08df3f --api-key api-password
```

```bash
py plot_pareto_results.py --expid 695d27d1f2ebb0848d08df3f --api-key api-password
```

**Server**

```bash
python3 -m plot_pareto_3obj --expid 695d845556363c92c2d4f888 --api-key api-password --api-base http://localhost:8198/api/v1
```

```bash
python3 -m hv_gd_3obj --expid 695d845556363c92c2d4f888 --api-key api-password --ref-point 3000000 3500 -2500 --api-base http://localhost:8198/api/v1
```

```bash
python3 -m plot_parallel_coords --expid 695d845556363c92c2d4f888 --api-key api-password --api-base http://localhost:8198/api/v1
```

```bash
py plot_pareto_results.py --expid 695d27d1f2ebb0848d08df3f --api-key api-password --api-base http://localhost:8198/api/v1
```