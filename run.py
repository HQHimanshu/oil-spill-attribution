#!/usr/bin/env python3
"""
OceanGuard AI — Unified System Bootloader & Launcher (SIH26143).
Boots both the FastAPI Backend and the Web Dashboard with automated health checks
and launches the browser automatically.
"""
import os
import sys
import time
import socket
import webbrowser
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def print_banner():
    banner = """
 ==============================================================================
  ██████╗  ██████╗███████╗ █████╗ ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ 
 ██╔═══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗
 ██║   ██║██║     █████╗  ███████║██╔██╗ ██║██║  ███╗██║   ██║███████║██████╔╝
 ██║   ██║██║     ██╔══╝  ██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔══██║██╔══██╗
 ╚██████╔╝╚██████╗███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║██║  ██║
  ╚═════╝  ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
              Autonomous Maritime Oil Spill Attribution & RAG System
 ==============================================================================
"""
    print(banner)

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def check_environment():
    print("[BOOT 1/4] Performing Pre-Flight Diagnostics...")
    print(f"  [+] Project Root   : {PROJECT_ROOT}")
    print(f"  [+] Python Runtime : {sys.version.split()[0]} ({sys.executable})")
    
    # Check key models
    seg_model = PROJECT_ROOT / "ml" / "models" / "sar_spill_segmentation_model.joblib"
    loc_model = PROJECT_ROOT / "ml" / "models" / "spill_location_model.joblib"
    
    if seg_model.exists():
        print(f"  [+] SAR Model      : READY ({round(seg_model.stat().st_size / (1024*1024), 2)} MB)")
    else:
        print("  [!] SAR Model      : Missing. Please run 'python ml/train_all.py'")
        
    if loc_model.exists():
        print(f"  [+] Location Model : READY ({round(loc_model.stat().st_size / (1024*1024), 2)} MB)")
    else:
        print("  [!] Location Model : Missing. Please run 'python ml/train_all.py'")

def start_backend():
    print("\n[BOOT 2/4] Booting FastAPI Backend Intelligence Server on http://127.0.0.1:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

def start_frontend():
    print("[BOOT 3/4] Booting Frontend Static Web Server on http://127.0.0.1:3000 ...")
    cmd = [sys.executable, "-m", "http.server", "3000", "--directory", "frontend"]
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

def main():
    print_banner()
    check_environment()
    
    # 1. Start Backend
    backend_proc = None
    if is_port_in_use(8000):
        print("  [i] Port 8000 already active (Reusing existing Backend instance)")
    else:
        backend_proc = start_backend()
        # Wait for backend to be ready
        for _ in range(20):
            if is_port_in_use(8000):
                print("  [+] Backend API is ONLINE and HEALTHY!")
                break
            time.sleep(0.5)

    # 2. Start Frontend
    frontend_proc = None
    if is_port_in_use(3000):
        print("  [i] Port 3000 already active (Reusing existing Frontend instance)")
    else:
        frontend_proc = start_frontend()
        for _ in range(10):
            if is_port_in_use(3000):
                print("  [+] Frontend Dashboard is ONLINE!")
                break
            time.sleep(0.4)

    # 3. Open Browser
    print("\n[BOOT 4/4] Launching OceanGuard AI Dashboard in default browser...")
    url = "http://127.0.0.1:3000"
    print(f"  >>> Access URL : {url}")
    print(f"  >>> Swagger API: http://127.0.0.1:8000/docs")
    print("\n" + "=" * 78)
    print("  [SYSTEM READY] Press Ctrl+C at any time to cleanly shut down all servers.")
    print("=" * 78 + "\n")
    
    time.sleep(0.8)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Terminating OceanGuard AI services...")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("[SHUTDOWN] All servers stopped cleanly. Goodbye!\n")

if __name__ == "__main__":
    main()
