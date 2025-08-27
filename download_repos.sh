#!/bin/bash
#
# Qwen2.5代码训练仓库批量下载脚本 (Shell版本)
# 支持并行下载、断点续传、进度显示
#

set -euo pipefail

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/qwen25_code_training_repositories.json"
OUTPUT_DIR="${SCRIPT_DIR}/repositories"
MAX_PARALLEL=4
RETRY_COUNT=3
DEPTH=1

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志文件
LOG_FILE="${SCRIPT_DIR}/repo_download_shell.log"

# 计数器
TOTAL_REPOS=0
SUCCESS_COUNT=0
FAILED_COUNT=0
declare -a FAILED_REPOS

# 打印带颜色的消息
print_msg() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

log_msg() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# 显示帮助信息
show_help() {
    cat << EOF
Qwen2.5代码训练仓库批量下载脚本

用法: $0 [选项]

选项:
    -h, --help              显示帮助信息
    -c, --config FILE       指定配置文件 (默认: qwen25_code_training_repositories.json)
    -o, --output DIR        指定输出目录 (默认: repositories)
    -j, --parallel N        并行下载数量 (默认: 4)
    -d, --depth N           Git克隆深度 (默认: 1)
    -p, --priority LEVEL    只下载指定优先级 (HIGH/MEDIUM)
    -l, --language LANG     只下载指定语言 (Python/Java/C++/Go/Rust)
    --dry-run              试运行，只显示要下载的仓库
    --resume               恢复中断的下载
    --cleanup              下载后清理不需要的文件

示例:
    $0                                    # 下载所有仓库
    $0 -p HIGH -j 8                      # 只下载高优先级，使用8个并行
    $0 -l Python -o /tmp/python_repos    # 只下载Python仓库到指定目录
    $0 --dry-run                         # 查看要下载的仓库列表
EOF
}

# 检查依赖
check_dependencies() {
    local missing_deps=()
    
    if ! command -v git >/dev/null 2>&1; then
        missing_deps+=("git")
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        missing_deps+=("jq")
    fi
    
    if ! command -v parallel >/dev/null 2>&1 && [ "$MAX_PARALLEL" -gt 1 ]; then
        print_msg "$YELLOW" "警告: GNU parallel未安装，将使用串行下载"
        MAX_PARALLEL=1
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_msg "$RED" "错误: 缺少依赖程序: ${missing_deps[*]}"
        print_msg "$YELLOW" "请安装缺少的依赖:"
        for dep in "${missing_deps[@]}"; do
            case "$dep" in
                git) echo "  - Ubuntu/Debian: sudo apt-get install git" ;;
                jq) echo "  - Ubuntu/Debian: sudo apt-get install jq" ;;
            esac
        done
        exit 1
    fi
}

# 解析JSON配置文件
parse_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_msg "$RED" "错误: 配置文件 $CONFIG_FILE 不存在"
        exit 1
    fi
    
    # 验证JSON格式
    if ! jq empty "$CONFIG_FILE" 2>/dev/null; then
        print_msg "$RED" "错误: 配置文件格式不正确"
        exit 1
    fi
    
    TOTAL_REPOS=$(jq '.repositories | length' "$CONFIG_FILE")
    log_msg "从配置文件加载了 $TOTAL_REPOS 个仓库"
}

# 获取仓库列表
get_repo_list() {
    local filter=""
    
    # 应用优先级过滤
    if [ -n "$PRIORITY_FILTER" ]; then
        filter+=" | select(.priority == \"$PRIORITY_FILTER\")"
    fi
    
    # 应用语言过滤  
    if [ -n "$LANGUAGE_FILTER" ]; then
        filter+=" | select(.language | ascii_downcase == \"$(echo "$LANGUAGE_FILTER" | tr '[:upper:]' '[:lower:]')\")"
    fi
    
    jq -r ".repositories[] $filter | \"\(.full_name)|\(.url)|\(.language)|\(.priority)|\(.name)\"" "$CONFIG_FILE"
}

# 获取仓库本地路径
get_repo_path() {
    local full_name=$1
    local language=$2
    local priority=$3
    local name=$4
    
    local priority_dir
    if [ "$priority" = "HIGH" ]; then
        priority_dir="high_priority"
    else
        priority_dir="medium_priority"
    fi
    
    echo "$OUTPUT_DIR/$priority_dir/$(echo "$language" | tr '[:upper:]' '[:lower:]')/$name"
}

# 检查仓库是否已存在
repo_exists() {
    local repo_path=$1
    [ -d "$repo_path/.git" ]
}

# 清理仓库文件
cleanup_repo() {
    local repo_path=$1
    
    if [ "$CLEANUP" != "true" ]; then
        return 0
    fi
    
    log_msg "清理仓库: $repo_path"
    
    # 删除大文件和不必要的目录
    find "$repo_path" -type d \( \
        -name "node_modules" -o \
        -name "__pycache__" -o \
        -name ".pytest_cache" -o \
        -name "target" -o \
        -name "build" -o \
        -name "dist" \
    \) -exec rm -rf {} + 2>/dev/null || true
    
    # 删除特定文件类型
    find "$repo_path" \( \
        -name "*.pyc" -o \
        -name "*.pyo" -o \
        -name "*.so" -o \
        -name "*.dylib" -o \
        -name "*.dll" \
    \) -delete 2>/dev/null || true
    
    # 压缩.git目录
    if [ -d "$repo_path/.git" ]; then
        cd "$repo_path"
        git gc --aggressive --prune=now 2>/dev/null || true
        cd - >/dev/null
    fi
}

# 下载单个仓库
clone_single_repo() {
    local repo_info=$1
    IFS='|' read -r full_name url language priority name <<< "$repo_info"
    
    local repo_path
    repo_path=$(get_repo_path "$full_name" "$language" "$priority" "$name")
    
    # 检查是否已存在
    if repo_exists "$repo_path"; then
        print_msg "$GREEN" "✓ $full_name 已存在，跳过"
        return 0
    fi
    
    # 创建目录
    mkdir -p "$(dirname "$repo_path")"
    
    # 开始下载
    print_msg "$BLUE" "⬇️  正在下载: $full_name"
    
    local attempt=1
    while [ $attempt -le $RETRY_COUNT ]; do
        if [ $attempt -gt 1 ]; then
            print_msg "$YELLOW" "重试 $full_name (第 $attempt 次)"
            sleep $((attempt - 1))
        fi
        
        # 执行git clone
        if git clone --depth="$DEPTH" --single-branch "$url" "$repo_path" 2>&1 | \
           sed "s/^/$name: /" | tee -a "$LOG_FILE"; then
            
            print_msg "$GREEN" "✅ $full_name 下载完成"
            
            # 清理文件
            cleanup_repo "$repo_path"
            
            # 获取仓库统计信息
            local file_count
            file_count=$(find "$repo_path" -type f | wc -l)
            local size_mb
            size_mb=$(du -sm "$repo_path" 2>/dev/null | cut -f1 || echo "0")
            
            log_msg "$full_name: $file_count 个文件, ${size_mb}MB"
            return 0
        else
            print_msg "$RED" "❌ $full_name 下载失败 (尝试 $attempt/$RETRY_COUNT)"
            rm -rf "$repo_path" 2>/dev/null || true
        fi
        
        ((attempt++))
    done
    
    print_msg "$RED" "❌ $full_name 最终失败"
    echo "$full_name|$url" >> "${SCRIPT_DIR}/failed_repos.txt"
    return 1
}

# 并行下载仓库
download_repos_parallel() {
    local repo_list=$1
    local repo_count
    repo_count=$(echo "$repo_list" | wc -l)
    
    if [ "$repo_count" -eq 0 ]; then
        print_msg "$YELLOW" "没有符合条件的仓库"
        return 0
    fi
    
    print_msg "$BLUE" "开始下载 $repo_count 个仓库 (并行数: $MAX_PARALLEL)"
    
    # 清空失败记录
    rm -f "${SCRIPT_DIR}/failed_repos.txt"
    
    if [ "$MAX_PARALLEL" -gt 1 ] && command -v parallel >/dev/null 2>&1; then
        # 使用GNU parallel
        echo "$repo_list" | parallel -j "$MAX_PARALLEL" --bar clone_single_repo {}
    else
        # 串行下载
        local current=0
        while IFS= read -r repo_info; do
            ((current++))
            print_msg "$BLUE" "进度: $current/$repo_count"
            if clone_single_repo "$repo_info"; then
                ((SUCCESS_COUNT++))
            else
                ((FAILED_COUNT++))
                FAILED_REPOS+=("$repo_info")
            fi
        done <<< "$repo_list"
    fi
    
    # 统计结果
    if [ -f "${SCRIPT_DIR}/failed_repos.txt" ]; then
        FAILED_COUNT=$(wc -l < "${SCRIPT_DIR}/failed_repos.txt")
        readarray -t FAILED_REPOS < "${SCRIPT_DIR}/failed_repos.txt"
    fi
    SUCCESS_COUNT=$((repo_count - FAILED_COUNT))
}

# 生成下载报告
generate_report() {
    local total=$((SUCCESS_COUNT + FAILED_COUNT))
    local success_rate=0
    
    if [ $total -gt 0 ]; then
        success_rate=$(( (SUCCESS_COUNT * 100) / total ))
    fi
    
    local report_file="${SCRIPT_DIR}/download_report_$(date +%s).txt"
    
    {
        echo "Qwen2.5代码训练仓库下载报告"
        echo "================================"
        echo "下载时间: $(date)"
        echo "总仓库数: $total"
        echo "成功下载: $SUCCESS_COUNT"
        echo "下载失败: $FAILED_COUNT"
        echo "成功率: ${success_rate}%"
        echo ""
        echo "输出目录: $OUTPUT_DIR"
        echo ""
        
        if [ ${#FAILED_REPOS[@]} -gt 0 ]; then
            echo "失败的仓库:"
            printf "%s\n" "${FAILED_REPOS[@]}" | cut -d'|' -f1 | sed 's/^/  - /'
            echo ""
        fi
        
        echo "目录结构:"
        if [ -d "$OUTPUT_DIR" ]; then
            tree "$OUTPUT_DIR" -d -L 3 2>/dev/null || find "$OUTPUT_DIR" -type d | head -20
        fi
        
    } > "$report_file"
    
    # 显示摘要
    print_msg "$GREEN" ""
    print_msg "$GREEN" "📊 下载完成报告"
    print_msg "$GREEN" "================================"
    print_msg "$GREEN" "总仓库数: $total"
    print_msg "$GREEN" "成功下载: $SUCCESS_COUNT ✅"
    if [ $FAILED_COUNT -gt 0 ]; then
        print_msg "$RED" "下载失败: $FAILED_COUNT ❌"
    fi
    print_msg "$GREEN" "成功率: ${success_rate}%"
    print_msg "$BLUE" "详细报告: $report_file"
    
    # 显示磁盘使用
    if [ -d "$OUTPUT_DIR" ]; then
        local total_size
        total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "未知")
        print_msg "$GREEN" "💾 总下载大小: $total_size"
    fi
}

# 恢复下载
resume_download() {
    if [ -f "${SCRIPT_DIR}/failed_repos.txt" ]; then
        print_msg "$YELLOW" "发现失败仓库记录，恢复下载..."
        local failed_list
        failed_list=$(cat "${SCRIPT_DIR}/failed_repos.txt" | cut -d'|' -f1)
        
        # 从原配置中重新获取这些仓库的信息
        local resume_list=""
        while IFS= read -r failed_repo; do
            local repo_info
            repo_info=$(jq -r ".repositories[] | select(.full_name == \"$failed_repo\") | \"\(.full_name)|\(.url)|\(.language)|\(.priority)|\(.name)\"" "$CONFIG_FILE")
            if [ -n "$repo_info" ]; then
                resume_list+="$repo_info"$'\n'
            fi
        done <<< "$failed_list"
        
        if [ -n "$resume_list" ]; then
            download_repos_parallel "$resume_list"
        fi
    else
        print_msg "$YELLOW" "没有找到失败的下载记录"
    fi
}

# 主函数
main() {
    local PRIORITY_FILTER=""
    local LANGUAGE_FILTER=""
    local DRY_RUN=false
    local RESUME=false
    local CLEANUP=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -j|--parallel)
                MAX_PARALLEL="$2"
                shift 2
                ;;
            -d|--depth)
                DEPTH="$2"
                shift 2
                ;;
            -p|--priority)
                PRIORITY_FILTER="$2"
                shift 2
                ;;
            -l|--language)
                LANGUAGE_FILTER="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --cleanup)
                CLEANUP=true
                shift
                ;;
            *)
                print_msg "$RED" "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 初始化
    print_msg "$BLUE" "🚀 Qwen2.5代码训练仓库批量下载器 (Shell版本)"
    print_msg "$BLUE" "=================================================="
    
    # 检查依赖
    check_dependencies
    
    # 解析配置
    parse_config
    
    # 恢复模式
    if [ "$RESUME" = true ]; then
        resume_download
        generate_report
        return $?
    fi
    
    # 获取仓库列表
    local repo_list
    repo_list=$(get_repo_list)
    local filtered_count
    filtered_count=$(echo "$repo_list" | wc -l)
    
    if [ -z "$repo_list" ]; then
        print_msg "$YELLOW" "没有符合条件的仓库"
        exit 0
    fi
    
    print_msg "$BLUE" "📦 找到 $filtered_count 个符合条件的仓库"
    
    if [ -n "$PRIORITY_FILTER" ]; then
        print_msg "$BLUE" "🔍 优先级过滤: $PRIORITY_FILTER"
    fi
    
    if [ -n "$LANGUAGE_FILTER" ]; then
        print_msg "$BLUE" "🔍 语言过滤: $LANGUAGE_FILTER"  
    fi
    
    print_msg "$BLUE" "💾 输出目录: $OUTPUT_DIR"
    print_msg "$BLUE" "🔄 并行数: $MAX_PARALLEL"
    print_msg "$BLUE" "📏 克隆深度: $DEPTH"
    
    # 试运行模式
    if [ "$DRY_RUN" = true ]; then
        print_msg "$YELLOW" "🔍 试运行模式 - 将要下载的仓库:"
        echo "$repo_list" | while IFS='|' read -r full_name url language priority name; do
            print_msg "$YELLOW" "  - $full_name ($language, $priority)"
        done
        exit 0
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"/{high_priority,medium_priority}/{python,java,c++,go,rust}
    
    # 导出函数以供parallel使用
    export -f clone_single_repo get_repo_path repo_exists cleanup_repo print_msg log_msg
    export OUTPUT_DIR DEPTH RETRY_COUNT LOG_FILE CLEANUP RED GREEN YELLOW BLUE NC
    
    # 开始下载
    local start_time
    start_time=$(date +%s)
    
    download_repos_parallel "$repo_list"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_msg "$BLUE" "⏱️  总耗时: ${duration} 秒"
    
    # 生成报告
    generate_report
    
    # 返回状态码
    if [ $FAILED_COUNT -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# 只在直接执行时运行main函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi