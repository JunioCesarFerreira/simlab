# Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [v1.0.0] – Initial Release (2025-10-12)

### Overview
This is the **first public and functional release** of the **SimLab** project — a distributed framework for managing and executing large-scale multi-objective simulations using Dockerized environments and a REST API interface.

The current version provides a **fully operational system** capable of running and monitoring experiments end-to-end through container orchestration and MongoDB integration.

### Features
- ✅ **Functional base architecture**
  - Master-node orchestrator for simulation execution (via SSH/SCP)
  - MO-engine (multi-objective optimization loop)
  - REST API for experiment management
  - MongoDB integration for experiment and generation tracking
- 🐳 **Dockerized environment**
  - Ready-to-run Docker Compose configurations for local and distributed setups  
  - Debug environments under `debug/` for simple testing or small experiments
- ⚙️ **Synthetic data mode**
  - Built-in synthetic benchmark evaluation (`DTLZ2`, `ZDT1`, `SCH1`) for validation and algorithm testing without running Cooja simulations
- 📡 **Asynchronous orchestration**
  - Multi-threaded simulation queue, automatic enqueue of waiting experiments
- 📁 **GridFS-based file management**
  - Storage and retrieval of simulation inputs, outputs, logs, and CSV results

### Documentation and Improvements (Planned)
This initial release is functional but not yet fully documented.  
The following enhancements are planned for upcoming versions:

- Complete documentation of setup and deployment workflows
- Additional testing and CI automation  
- Extended examples of experiment submission and monitoring  
- Benchmark dataset publication and performance validation  
- Development of a graphical user interface (GUI) in Vue.js to simplify experiment configuration, execution monitoring, and result visualization  
- English and Portuguese documentation parity  
- More algorithms and analyzers

### Notes
This version establishes the operational baseline of SimLab.  
Subsequent versions will focus on documentation, reproducibility, and academic publication preparation.

---

**Author:**  
Junio Cesar Ferreira<br/>
Institute of Mathematical and Computer Sciences (ICMC), University of São Paulo (USP)
