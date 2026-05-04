"""
一键启动脚本：同时启动 FastAPI 后端 + Streamlit 前端。

用法：
    python start.py

启动后：
    - 后端：http://127.0.0.1:8000
    - 前端：http://localhost:8501

按 Ctrl+C 同时关闭两个服务。
"""

from __future__ import annotations

import glob
import subprocess
import sys
import time
import signal
import os

import httpx
import yaml


def load_app_config(project_dir: str) -> dict:
    """读取 config/app.yml 配置。"""
    config_path = os.path.join(project_dir, "config", "app.yml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def cleanup_temp_files(upload_dir: str = "tmp_uploads", max_age_hours: int = 24):
    """删除超过 max_age_hours 小时的临时上传文件。"""
    if not os.path.isdir(upload_dir):
        return
    now = time.time()
    cutoff = now - max_age_hours * 3600
    removed = 0
    for path in glob.glob(os.path.join(upload_dir, "**", "*"), recursive=True):
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  已清理 {removed} 个过期临时文件")


def wait_for_backend(host: str = "127.0.0.1", port: int = 8000, timeout: float = 30.0):
    """轮询后端 /api/health 直到就绪，最多等待 timeout 秒。"""
    url = f"http://{host}:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    # 获取当前脚本所在目录，确保子进程在正确的工作目录下启动
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_app_config(project_dir)
    backend_host = cfg.get("backend_host", "127.0.0.1")
    backend_port = cfg.get("backend_port", 8000)
    frontend_port = cfg.get("frontend_port", 8501)

    print("=" * 50)
    print("  AI 面试准备助手 —— 一键启动")
    print("=" * 50)

    # 启动前清理过期临时文件
    upload_dir = os.path.join(project_dir, cfg.get("temp_upload_dir", "tmp_uploads"))
    max_age = cfg.get("temp_file_max_age_hours", 24)
    cleanup_temp_files(upload_dir, max_age)

    # 启动 FastAPI 后端
    print(f"\n[1/2] 正在启动 FastAPI 后端 (端口 {backend_port}) ...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app",
         "--host", backend_host, "--port", str(backend_port)],
        cwd=project_dir,
    )

    # 等待后端真正就绪（轮询健康检查接口）
    print("      等待后端就绪 ...", end="", flush=True)
    if wait_for_backend(host=backend_host, port=backend_port):
        print(" 就绪！")
    else:
        print(" 超时！后端启动可能失败，请检查日志。")
        backend.terminate()
        sys.exit(1)

    # 启动 Streamlit 前端
    print(f"[2/2] 正在启动 Streamlit 前端 (端口 {frontend_port}) ...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(frontend_port)],
        cwd=project_dir,
    )

    print("\n" + "=" * 50)
    print("  启动完成！")
    print(f"  后端 API：http://{backend_host}:{backend_port}")
    print(f"  前端页面：http://localhost:{frontend_port}")
    print("  按 Ctrl+C 关闭所有服务")
    print("=" * 50 + "\n")

    # 注册信号处理：Ctrl+C 时同时终止两个子进程
    def shutdown(sig, frame):
        print("\n正在关闭服务 ...")
        backend.terminate()
        frontend.terminate()
        backend.wait(timeout=5)
        frontend.wait(timeout=5)
        print("已关闭。")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 等待任一子进程退出
    try:
        frontend.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
