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
py plot_pareto_results.py --expid 69610399efa1000afe8a4f85 --api-key api-password --keep-the-files True
```

**Server**

Replece `secret` with your api key.

```bash
python3 -m plot_pareto_3obj --expid 695d845556363c92c2d4f888 --api-key secret --api-base http://localhost:8198/api/v1
```

```bash
python3 -m hv_gd_3obj --expid 695d845556363c92c2d4f888 --api-key secret --ref-point 3000000 3500 -2500 --api-base http://localhost:8198/api/v1
```

```bash
python3 -m plot_parallel_coords --expid 695d845556363c92c2d4f888 --api-key secret --api-base http://localhost:8198/api/v1
```

```bash
python3 plot_pareto_results.py --expid 695fbdac3be5f623aa9af752 --api-key secret --api-base http://localhost:8198/api/v1
```


**Local/Server**

```bash
py plot_pareto_results.py --expid 695fba4ba8bc65564cea8697 --keep-the-files --api-key secret --api-base http://andromeda.lasdpc.icmc.usp.br/:8198/api/v1
```