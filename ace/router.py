"""
ace/router.py - APIRouter for ACE Proctoring Engine integration.
Allows mounting proctoring endpoints directly into the main VeriProof FastAPI app.
"""

import os
import asyncio
from pathlib import Path
from typing import Set, List, Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from ace.config import Config
from ace.core.engine import ProctorEngine
from ace.core.system_guard import SystemGuard
from ace.core.code_analyzer import CodeIntegrityAnalyzer

proctor_router = APIRouter(tags=["Proctoring & Anti-Cheat"])

# Global Instances
engine = ProctorEngine()
system_guard = SystemGuard()
code_analyzer = CodeIntegrityAnalyzer()

active_websockets: Set[WebSocket] = set()
ws_lock = asyncio.Lock()
telemetry_broadcast_task: Optional[asyncio.Task] = None


class KeystrokePayload(BaseModel):
    event_type: str = "key"
    code_length: int = 0
    chars_added: int = 1
    key: str = ""
    timestamp_ms: Optional[float] = None


class TabSwitchPayload(BaseModel):
    event_type: str = "blur"
    details: str = ""


class CodeSubmissionPayload(BaseModel):
    source_code: str = ""
    language: str = "python"


def on_violation_event(event_data: Dict[str, Any]):
    """Bridge synchronous violation logger events into asyncio event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _broadcast():
        async with ws_lock:
            disconnected = set()
            for ws in active_websockets:
                try:
                    await ws.send_json(event_data)
                except Exception:
                    disconnected.add(ws)
            for dead_ws in disconnected:
                active_websockets.discard(dead_ws)

    asyncio.run_coroutine_threadsafe(_broadcast(), loop)


async def _periodic_telemetry_broadcaster():
    """Periodically broadcasts latest telemetry stats (10Hz) to all WebSocket clients."""
    while True:
        try:
            await asyncio.sleep(0.1)
            telemetry = engine.get_telemetry()
            if active_websockets:
                message = {
                    "event": "telemetry",
                    "data": telemetry,
                }
                async with ws_lock:
                    dead = set()
                    for ws in active_websockets:
                        try:
                            await ws.send_json(message)
                        except Exception:
                            dead.add(ws)
                    for d in dead:
                        active_websockets.discard(d)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.2)


def start_proctor_engine():
    """Start proctoring engine and background workers."""
    global telemetry_broadcast_task
    print("[ACE Proctor] Starting hardware camera and AI proctoring engine...")
    engine.logger.subscribe(on_violation_event)
    engine.start()
    try:
        telemetry_broadcast_task = asyncio.create_task(_periodic_telemetry_broadcaster())
    except RuntimeError:
        pass
    print("[ACE Proctor] Engine ready on /api/stream and /ws/telemetry.")


def stop_proctor_engine():
    """Clean up and release all camera and hardware resources."""
    global telemetry_broadcast_task
    print("[ACE Proctor] Releasing camera and proctoring workers...")
    if telemetry_broadcast_task:
        telemetry_broadcast_task.cancel()
    engine.logger.unsubscribe(on_violation_event)
    engine.stop()
    print("[ACE Proctor] Shutdown complete.")


@proctor_router.get("/api/stream")
def video_stream():
    """Live MJPEG Video Stream with AI Bounding Boxes and HUD."""
    def frame_generator():
        while True:
            jpeg_bytes = engine.get_latest_jpeg()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            else:
                import time
                time.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@proctor_router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time bi-directional telemetry and violation alerts WebSocket feed."""
    await websocket.accept()
    async with ws_lock:
        active_websockets.add(websocket)

    try:
        await websocket.send_json({
            "event": "telemetry",
            "data": engine.get_telemetry(),
        })

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            elif data == "recalibrate":
                success = engine.recalibrate(duration_sec=3)
                await websocket.send_json({
                    "event": "recalibration_result",
                    "success": success,
                })
    except WebSocketDisconnect:
        pass
    finally:
        async with ws_lock:
            active_websockets.discard(websocket)


@proctor_router.get("/api/status")
@proctor_router.get("/api/proctor/status")
def get_status():
    """REST endpoint for proctoring status polling."""
    return {
        "status": "running" if engine._running else "stopped",
        "telemetry": engine.get_telemetry(),
        "recent_violations": engine.logger.get_recent_violations(limit=10),
    }


@proctor_router.get("/api/system/audit")
def audit_system():
    """Audits connected display count and scans for prohibited background apps."""
    return system_guard.audit_environment()


@proctor_router.post("/api/recalibrate")
def trigger_recalibration():
    """Trigger dynamic head pose and eye gaze recalibration."""
    success = engine.recalibrate(duration_sec=3)
    return {"success": success, "baseline": engine.baseline}


@proctor_router.get("/api/screenshots/{filename}")
def get_screenshot(filename: str):
    """Retrieve saved violation proof screenshot."""
    filepath = Path(Config.SCREENSHOT_DIR) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Screenshot evidence not found")
    return FileResponse(filepath, media_type="image/jpeg")


@proctor_router.post("/api/exam/keystroke")
def record_keystroke(payload: KeystrokePayload):
    """Ingests typing dynamics and checks for copy-paste bursts."""
    return code_analyzer.record_keystroke(
        event_type=payload.event_type,
        code_length=payload.code_length,
        chars_added=payload.chars_added,
        key=payload.key,
        timestamp_ms=payload.timestamp_ms,
    )


@proctor_router.post("/api/exam/tab_switch")
def record_tab_switch(payload: TabSwitchPayload):
    """Records browser blur and tab-switching strikes."""
    return code_analyzer.record_tab_switch(event_type=payload.event_type, details=payload.details)


@proctor_router.post("/api/exam/submit")
def evaluate_exam_submission(payload: CodeSubmissionPayload):
    """Performs AST structural syntax analysis on code submission."""
    return code_analyzer.evaluate_submission(
        source_code=payload.source_code,
        language=payload.language,
    )
