# Qwen2.5代码训练仓库自动下载工具

为Qwen2.5大模型代码训练精选的25个高质量GitHub仓库自动下载工具，支持Python和Shell两种实现方式。

## 🎯 项目特色

- **精选高质量仓库**: 25个精心筛选的开源项目，涵盖AI/ML、分布式系统、数据库、Web框架
- **智能下载管理**: 支持并行下载、断点续传、进度显示、自动重试
- **灵活过滤选项**: 支持按语言、优先级、自定义条件过滤下载
- **完整报告生成**: 详细的下载报告和统计信息
- **跨平台支持**: Python和Shell两种实现，适配不同环境

## 📦 仓库清单概览

| 语言 | 仓库数量 | 重点项目 |
|------|----------|----------|
| **Python** | 6个 | TensorFlow, PyTorch, HuggingFace Transformers, Django, FastAPI |
| **Java** | 5个 | Spring Framework, Elasticsearch, Apache Kafka, Apache Hadoop |
| **C++** | 5个 | Facebook Folly, Google LevelDB, Microsoft vcpkg, osquery |
| **Go** | 5个 | Kubernetes, Docker, Prometheus, etcd, Istio |
| **Rust** | 4个 | Rust语言本身, Tauri, SurrealDB, Tokio |

总计：**25个仓库**，预估总下载大小：**~2-5GB**

## 🚀 快速开始

### 方式一：Python版本（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载所有仓库
python repo_downloader.py

# 3. 只下载高优先级Python仓库
python repo_downloader.py --priority HIGH --language Python

# 4. 试运行查看将要下载的仓库
python repo_downloader.py --dry-run
```

### 方式二：Shell版本

```bash
# 1. 设置执行权限
chmod +x download_repos.sh

# 2. 下载所有仓库
./download_repos.sh

# 3. 只下载Go语言仓库，使用8个并行
./download_repos.sh --language Go --parallel 8

# 4. 恢复中断的下载
./download_repos.sh --resume
```

## 📋 使用说明

### Python版本参数

```bash
python repo_downloader.py [选项]

选项:
  -h, --help              显示帮助信息
  -c, --config FILE       配置文件路径 (默认: qwen25_code_training_repositories.json)
  -o, --output DIR        输出目录 (默认: repositories)  
  -w, --workers N         并发下载数 (默认: 4)
  -p, --priority LEVEL    只下载指定优先级 (HIGH/MEDIUM)
  -l, --language LANG     只下载指定语言仓库
  --dry-run              试运行模式
```

### Shell版本参数

```bash
./download_repos.sh [选项]

选项:
  -h, --help              显示帮助信息
  -c, --config FILE       配置文件路径
  -o, --output DIR        输出目录
  -j, --parallel N        并发下载数 (默认: 4)
  -d, --depth N           Git克隆深度 (默认: 1)
  -p, --priority LEVEL    只下载指定优先级
  -l, --language LANG     只下载指定语言
  --dry-run              试运行模式
  --resume               恢复下载
  --cleanup              下载后清理
```

## 📁 目录结构

下载完成后的目录结构：

```
repositories/
├── high_priority/          # 高优先级项目
│   ├── python/            # Python项目
│   │   ├── tensorflow/
│   │   ├── pytorch/
│   │   └── transformers/
│   ├── java/              # Java项目
│   │   ├── spring-framework/
│   │   └── elasticsearch/
│   ├── cpp/               # C++项目
│   ├── go/                # Go项目
│   │   ├── kubernetes/
│   │   └── prometheus/
│   └── rust/              # Rust项目
│       └── rust/
├── medium_priority/        # 中优先级项目
│   └── ...
└── logs/                  # 日志和报告
    ├── download_report_*.json
    └── download_report_*.txt
```

## 🛠 系统要求

### Python版本
- Python 3.7+
- Git 2.0+
- 依赖包：`tqdm`, `requests`

### Shell版本  
- Bash 4.0+
- Git 2.0+
- jq (JSON处理工具)
- GNU parallel (可选，用于并行下载)

### 安装依赖

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install git jq parallel python3-pip
pip3 install -r requirements.txt
```

**macOS:**
```bash
brew install git jq parallel
pip3 install -r requirements.txt
```

**CentOS/RHEL:**
```bash
sudo yum install git epel-release
sudo yum install jq parallel python3-pip
pip3 install -r requirements.txt
```

## 📊 下载报告示例

```
📊 下载完成报告
==================================
总仓库数: 25
成功下载: 23 ✅
下载失败: 2 ❌
成功率: 92.0%
💾 总下载大小: 3.2 GB
报告保存至: repositories/logs/download_report_1698765432.json
```

## ⚙️ 高级配置

### 自定义过滤规则

编辑配置文件中的 `filtering_criteria` 部分：

```json
{
  "filtering_criteria": {
    "min_stars": 10000,
    "active_development": true,
    "code_quality": "high",
    "exclude_patterns": [
      "*/node_modules/*",
      "*/target/*",
      "*/.git/objects/pack/*.pack"
    ]
  }
}
```

### 环境变量配置

```bash
# 设置默认下载目录
export QWEN_REPO_DIR="/path/to/repositories"

# 设置默认并发数
export QWEN_PARALLEL_JOBS=8

# 设置Git克隆深度
export QWEN_GIT_DEPTH=1
```

## 🔍 故障排除

### 常见问题

1. **Git克隆失败**
   ```bash
   # 检查网络连接和Git配置
   git config --global http.postBuffer 524288000
   git config --global http.maxRequestBuffer 100M
   ```

2. **权限错误**
   ```bash
   # 确保有写入权限
   chmod -R 755 repositories/
   ```

3. **磁盘空间不足**
   ```bash
   # 检查可用空间
   df -h
   # 启用清理模式
   ./download_repos.sh --cleanup
   ```

4. **网络超时**
   ```bash
   # 设置Git超时
   git config --global http.lowSpeedLimit 1000
   git config --global http.lowSpeedTime 300
   ```

### 日志查看

```bash
# Python版本日志
tail -f repo_download.log

# Shell版本日志  
tail -f repo_download_shell.log
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个工具。

### 添加新仓库

1. 编辑 `qwen25_code_training_repositories.json`
2. 添加仓库信息，确保包含所有必需字段
3. 更新仓库总数和统计信息

### 报告Bug

请提供以下信息：
- 操作系统和版本
- Python/Bash版本
- 完整的错误信息
- 复现步骤

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有开源项目维护者的贡献，是他们的努力让我们能够构建这个高质量的代码训练数据集。

---

**注意**: 下载的代码仅用于机器学习研究和教育目的，请遵守各项目的开源许可证。