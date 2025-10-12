# Debug

In this directory, you'll find scripts that are useful for debugging.

- `requests`: These are preformatted JSON files for testing, requiring only ID adjustments to facilitate testing using the API's Swagger.

- `docker`: These are Docker files to facilitate preparing a local environment for debugging.

## Usage

To run SimLab locally using the preconfigured debug `docker-compose` setups. These are intended for testing, experimentation, and small-scale local execution.

### Prerequisites

- Docker and Docker Compose installed on your machine  
- (Optional) Ensure ports used in compose files do not conflict with existing services  

### 1. Choose a debug compose setup

Inside the `debug/` directory, you should find (for example):

- `minimal-infra`: This docker-compose can be used to generate a minimal structure for testing on localhost.
- `mongo-cooja`: Use this docker-compose to create a minimal local MongoDB and three Cooja containers, allowing local Python component execution for debugging and development.

### 2. Bring up the services

In a terminal, navigate to the root of the directory, and run:

```bash
cd debug/<<directory>>
```

You can add `--build` to force rebuilding, and `-d` to run in detached mode.

To view logs in real time, omit `-d`:
```bash
docker-compose up --build -d
```

### 3. Verify running containers

Run:

```bash
docker-compose ps
```

You should see containers like:

* `master_node`
* `mo_engine`
* `rest-api`
* `mongodb`
* `cooja1` (one or more simulation containers)

Check that their state is **Up** and the ports are mapped correctly.

### 4. Interact with the system

* Use the **REST API** (e.g., via browser, `curl`, or Postman) on the configured host port to submit experiments, view status, etc.
* If synthetic mode is enabled (via `ENABLE_DATA_SYNTHETIC`), results may be generated without launching real simulations.
* Monitor logs of individual containers:

```bash
docker-compose logs master-node
docker-compose logs mo-engine
```

### 5. Tear down / cleanup

When finished, stop and remove all containers:

```bash
docker-compose down
```

You may also include volumes clean-up if needed:

```bash
docker-compose down -v
```