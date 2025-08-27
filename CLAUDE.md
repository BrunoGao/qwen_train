# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a repository collection management system for Qwen2.5 code training, containing 25 curated high-quality repositories organized by priority and programming language. The system has successfully downloaded 1,669MB of production-ready source code across Python, Java, Go, Rust, and C++.

## Common Commands

### Repository Management
```bash
# Install dependencies
pip install -r requirements.txt

# Download all repositories
python repo_downloader.py

# Download specific language/priority
python repo_downloader.py --priority HIGH --language Python
python repo_downloader.py --dry-run

# Shell-based downloading with parallel support
chmod +x download_repos.sh
./download_repos.sh --language Go --parallel 8
./download_repos.sh --resume
```

### Working with Downloaded Projects

Each downloaded repository maintains its own development commands:

**Python projects** (6 repos: TensorFlow, PyTorch, HuggingFace, Django, FastAPI, Flask):
```bash
cd repositories/high_priority/python/<project>
pip install -e .
pytest  # or python -m pytest
```

**Java projects** (5 repos: Elasticsearch, Hadoop, Kafka, Spring Framework, Spring Boot):
```bash
cd repositories/high_priority/java/<project>
./gradlew build
./gradlew test
```

**Go projects** (5 repos: Kubernetes, Docker/Moby, Prometheus, etcd, Istio):
```bash
cd repositories/high_priority/go/<project>
go build
go test ./...
```

**Rust projects** (4 repos: Rust Language, SurrealDB, Tauri, Tokio):
```bash
cd repositories/high_priority/rust/<project>
cargo build
cargo test
```

**C++ projects** (5 repos: Folly, LevelDB, simdjson, osquery, vcpkg):
```bash
cd repositories/high_priority/c++/<project>
cmake . && make
ctest  # if tests are configured
```

## Architecture and Structure

### Directory Organization
- `repositories/high_priority/` - 19 critical projects (TensorFlow, Kubernetes, Rust language, etc.)
- `repositories/medium_priority/` - 6 secondary projects (Flask, Hadoop, Tauri, etc.)
- `repositories/logs/` - Download reports and execution logs

### Key Configuration Files
- `qwen25_code_training_repositories.json` - Central repository metadata and configuration
- `qwen25_code_training_repositories.csv` - Repository list in CSV format
- `requirements.txt` - Python dependencies (tqdm, requests)

### Repository Management Scripts
- `repo_downloader.py` - Main Python implementation with filtering and resume capabilities
- `download_repos.sh` - Shell script with parallel download support
- `download_qwen.py` - Alternative download implementation

## Understanding the Codebase

### Repository Selection Criteria
All repositories meet strict quality standards:
- 10,000+ GitHub stars minimum
- Code quality scores of 8.5-9.8
- Active development and maintenance
- High educational value for ML training

### Language Distribution
- **Python (683MB)**: ML/AI frameworks, web development
- **Java (700MB)**: Enterprise systems, big data processing  
- **Go (393MB)**: Container orchestration, distributed systems
- **Rust (231MB)**: System programming, async runtimes
- **C++ (81MB)**: High-performance libraries

### Monitoring and Status
Check download status and logs:
```bash
# View repository list
cat qwen25_code_training_repositories.csv

# Check download reports
cat repositories/logs/download_report_*.json

# Monitor specific language downloads
python repo_downloader.py --language Python --dry-run
```

## Working with Individual Projects

Each repository maintains its original structure and development practices. Key projects include:

- **TensorFlow** (`repositories/high_priority/python/tensorflow/`) - ML framework with Bazel build system
- **Kubernetes** (`repositories/high_priority/go/kubernetes/`) - Container orchestration with extensive Go modules
- **Spring Boot** (`repositories/high_priority/java/spring-boot/`) - Java web framework with Gradle build
- **Rust Language** (`repositories/high_priority/rust/rust/`) - Compiler with complex Cargo workspace
- **Folly** (`repositories/high_priority/c++/folly/`) - Facebook's C++ library with CMake build system

When working with individual projects, refer to their original documentation and build systems rather than the collection-level tools.