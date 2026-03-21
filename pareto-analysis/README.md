## Commands Examples

**Localhost**

```bash
py plot_pareto_results.py --expid 69b97e5c911986d1242e5a7e --api-key api-password --keep-the-files True
```

**Server**

Replece `secret` with your api key.

```bash
python3 plot_pareto_results.py --expid 69b97e5c911986d1242e5a7e --api-key secret --api-base http://localhost:8198/api/v1
```


**Local/Server**

```bash
py plot_pareto_results.py --expid 69b97e5c911986d1242e5a7e --keep-the-files --api-key secret --api-base http://andromeda.lasdpc.icmc.usp.br/:8198/api/v1
```