"""
VeriProof AI Proctoring Engine Service Launcher.
Starts the ACE High-Strictness Proctoring Microservice on port 8000.
"""

import sys
import os
from pathlib import Path
import uvicorn

# Ensure python_engine directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    port = int(os.getenv("PROCTOR_PORT", "8000"))
    print("\n" + "=" * 65)
    print("      VERIPROOF - High-Strictness AI Proctoring Engine")
    print("=" * 65)
    print(f"[Proctor] Live Video Stream: http://localhost:{port}/api/stream")
    print(f"[Proctor] Telemetry Feed:    ws://localhost:{port}/ws/telemetry")
    print(f"[Proctor] System Audit:      http://localhost:{port}/api/system/audit")
    print("=" * 65 + "\n")

    uvicorn.run(
        "ace.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
