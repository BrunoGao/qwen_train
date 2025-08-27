#!/usr/bin/env python3
"""
Qwen2.5代码训练仓库自动下载脚本
支持从JSON配置文件批量下载GitHub仓库，包含进度显示、断点续传、文件过滤等功能
"""

import json
import os
import sys
import subprocess
import shutil
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class RepoInfo:
    """仓库信息数据类"""
    id: int
    name: str
    full_name: str
    url: str
    language: str
    stars: str
    priority: str
    code_quality_score: float

class RepoDownloader:
    """仓库下载器主类"""
    
    def __init__(self, config_file: str = "qwen25_code_training_repositories.json", 
                 output_dir: str = "repositories", max_workers: int = 4):
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.setup_logging()
        self.setup_directories()
        self.repos = []
        self.failed_repos = []
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('repo_download.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """创建目录结构"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "high_priority").mkdir(exist_ok=True)
        (self.output_dir / "medium_priority").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
    def load_repositories(self) -> List[RepoInfo]:
        """加载仓库配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            repos = []
            for repo_data in data['repositories']:
                repo = RepoInfo(
                    id=repo_data['id'],
                    name=repo_data['name'],
                    full_name=repo_data['full_name'],
                    url=repo_data['url'],
                    language=repo_data['language'],
                    stars=repo_data['stars'],
                    priority=repo_data['priority'],
                    code_quality_score=repo_data['code_quality_score']
                )
                repos.append(repo)
            
            self.logger.info(f"成功加载 {len(repos)} 个仓库配置")
            return repos
            
        except FileNotFoundError:
            self.logger.error(f"配置文件 {self.config_file} 不存在")
            sys.exit(1)
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件格式错误: {e}")
            sys.exit(1)
            
    def check_git_available(self) -> bool:
        """检查Git是否可用"""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Git未安装或不可用，请先安装Git")
            return False
            
    def check_git_lfs_available(self) -> bool:
        """检查Git LFS是否可用"""
        try:
            subprocess.run(['git', 'lfs', 'version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Git LFS未安装，大文件可能无法正确下载")
            return False
            
    def get_repo_path(self, repo: RepoInfo) -> Path:
        """获取仓库本地路径"""
        priority_dir = "high_priority" if repo.priority == "HIGH" else "medium_priority"
        return self.output_dir / priority_dir / repo.language.lower() / repo.name
        
    def is_repo_exists(self, repo_path: Path) -> bool:
        """检查仓库是否已存在"""
        return repo_path.exists() and (repo_path / ".git").exists()
        
    def clone_repository(self, repo: RepoInfo, retry_count: int = 3) -> bool:
        """克隆单个仓库"""
        repo_path = self.get_repo_path(repo)
        
        # 检查是否已存在
        if self.is_repo_exists(repo_path):
            self.logger.info(f"仓库 {repo.full_name} 已存在，跳过下载")
            return True
            
        # 创建目录
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建git clone命令
        cmd = [
            'git', 'clone',
            '--depth=1',  # 浅克隆，节省空间和时间
            '--single-branch',
            repo.url,
            str(repo_path)
        ]
        
        for attempt in range(retry_count):
            try:
                self.logger.info(f"正在下载 {repo.full_name} (尝试 {attempt + 1}/{retry_count})")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时输出进度
                for line in process.stdout:
                    if "Receiving objects" in line or "Resolving deltas" in line:
                        print(f"\r{repo.name}: {line.strip()}", end="", flush=True)
                        
                process.wait()
                
                if process.returncode == 0:
                    print()  # 换行
                    self.logger.info(f"✅ {repo.full_name} 下载完成")
                    
                    # 清理不需要的文件
                    self.cleanup_repository(repo_path)
                    return True
                else:
                    self.logger.warning(f"❌ {repo.full_name} 下载失败，返回码: {process.returncode}")
                    
            except Exception as e:
                self.logger.error(f"下载 {repo.full_name} 时出错: {e}")
                
            # 清理失败的克隆
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
                
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # 指数退避
                
        return False
        
    def cleanup_repository(self, repo_path: Path):
        """清理仓库中不需要的文件"""
        cleanup_patterns = [
            "**/.git/objects/pack/*.pack",  # 大的pack文件
            "**/node_modules",
            "**/target/debug",
            "**/target/release", 
            "**/build",
            "**/dist",
            "**/*.pyc",
            "**/__pycache__",
            "**/test/fixtures/**/*.dat",
            "**/docs/images/**/*.png",
            "**/docs/images/**/*.jpg"
        ]
        
        for pattern in cleanup_patterns:
            for file_path in repo_path.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path, ignore_errors=True)
                except Exception as e:
                    self.logger.debug(f"清理文件 {file_path} 失败: {e}")
                    
    def get_repo_size(self, repo_path: Path) -> float:
        """获取仓库大小(MB)"""
        total_size = 0
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
        return total_size / (1024 * 1024)  # 转换为MB
        
    def download_repos_parallel(self, repos: List[RepoInfo]) -> Dict[str, List[RepoInfo]]:
        """并行下载仓库"""
        successful = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_repo = {
                executor.submit(self.clone_repository, repo): repo 
                for repo in repos
            }
            
            with tqdm(total=len(repos), desc="下载进度", unit="repo") as pbar:
                for future in as_completed(future_to_repo):
                    repo = future_to_repo[future]
                    try:
                        success = future.result()
                        if success:
                            successful.append(repo)
                        else:
                            failed.append(repo)
                    except Exception as e:
                        self.logger.error(f"下载 {repo.full_name} 时发生异常: {e}")
                        failed.append(repo)
                    finally:
                        pbar.update(1)
                        
        return {"successful": successful, "failed": failed}
        
    def generate_report(self, results: Dict[str, List[RepoInfo]]):
        """生成下载报告"""
        successful = results["successful"]
        failed = results["failed"]
        
        report = {
            "summary": {
                "total_repos": len(successful) + len(failed),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{len(successful)/(len(successful)+len(failed))*100:.1f}%",
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "successful_repos": [
                {
                    "name": repo.full_name,
                    "language": repo.language,
                    "priority": repo.priority,
                    "local_path": str(self.get_repo_path(repo)),
                    "size_mb": round(self.get_repo_size(self.get_repo_path(repo)), 2)
                } for repo in successful
            ],
            "failed_repos": [
                {
                    "name": repo.full_name,
                    "language": repo.language,
                    "priority": repo.priority,
                    "url": repo.url
                } for repo in failed
            ]
        }
        
        # 保存报告
        report_file = self.output_dir / "logs" / f"download_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 打印摘要
        print("\n" + "="*50)
        print("📊 下载完成报告")
        print("="*50)
        print(f"总仓库数: {report['summary']['total_repos']}")
        print(f"成功下载: {report['summary']['successful']} ✅")
        print(f"下载失败: {report['summary']['failed']} ❌") 
        print(f"成功率: {report['summary']['success_rate']}")
        print(f"报告保存至: {report_file}")
        
        if failed:
            print(f"\n❌ 失败的仓库:")
            for repo in failed:
                print(f"  - {repo.full_name}")
                
        total_size = sum(repo_data['size_mb'] for repo_data in report['successful_repos'])
        print(f"\n💾 总下载大小: {total_size:.1f} MB")
        
    def run(self, priority_filter: Optional[str] = None, language_filter: Optional[str] = None):
        """运行下载器"""
        print("🚀 Qwen2.5代码训练仓库自动下载器")
        print("="*50)
        
        # 检查依赖
        if not self.check_git_available():
            return False
            
        self.check_git_lfs_available()
        
        # 加载仓库配置
        all_repos = self.load_repositories()
        
        # 应用过滤器
        repos_to_download = all_repos
        if priority_filter:
            repos_to_download = [r for r in repos_to_download if r.priority == priority_filter]
            print(f"🔍 优先级过滤: {priority_filter}")
            
        if language_filter:
            repos_to_download = [r for r in repos_to_download if r.language.lower() == language_filter.lower()]
            print(f"🔍 语言过滤: {language_filter}")
            
        print(f"📦 准备下载 {len(repos_to_download)} 个仓库")
        print(f"💾 下载目录: {self.output_dir.absolute()}")
        print(f"🔄 并发数: {self.max_workers}")
        
        # 开始下载
        start_time = time.time()
        results = self.download_repos_parallel(repos_to_download)
        end_time = time.time()
        
        print(f"\n⏱️ 总耗时: {end_time - start_time:.1f} 秒")
        
        # 生成报告
        self.generate_report(results)
        
        return len(results["failed"]) == 0

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5代码训练仓库自动下载器")
    parser.add_argument("--config", "-c", default="qwen25_code_training_repositories.json",
                       help="配置文件路径 (默认: qwen25_code_training_repositories.json)")
    parser.add_argument("--output", "-o", default="repositories",
                       help="输出目录 (默认: repositories)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="并发下载数 (默认: 4)")
    parser.add_argument("--priority", "-p", choices=["HIGH", "MEDIUM"],
                       help="只下载指定优先级的仓库")
    parser.add_argument("--language", "-l", 
                       help="只下载指定语言的仓库 (Python/Java/C++/Go/Rust)")
    parser.add_argument("--dry-run", action="store_true",
                       help="试运行，只显示将要下载的仓库列表")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # 试运行模式
        downloader = RepoDownloader(args.config, args.output, args.workers)
        repos = downloader.load_repositories()
        
        if args.priority:
            repos = [r for r in repos if r.priority == args.priority]
        if args.language:
            repos = [r for r in repos if r.language.lower() == args.language.lower()]
            
        print(f"🔍 将要下载的仓库 ({len(repos)} 个):")
        for repo in repos:
            print(f"  - {repo.full_name} ({repo.language}, {repo.priority}, {repo.stars})")
        return
        
    # 正常运行
    downloader = RepoDownloader(args.config, args.output, args.workers)
    success = downloader.run(args.priority, args.language)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()