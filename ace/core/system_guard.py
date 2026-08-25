"""
ace/core/system_guard.py - Operating System & Environment Security Guard.
Audits running processes for cheating tools (Discord, AnyDesk, TeamViewer, OBS),
detects multi-monitor configurations, and flags suspicious background activity.
"""

import os
import sys
import ctypes
from typing import Dict, List, Any, Optional
import psutil
from ace.config import Config


class SystemGuard:
    """
    Scans the local host OS for prohibited software, unauthorized screen sharing,
    and multi-monitor setups that enable candidate cheating.
    """

    DEFAULT_PROHIBITED_PROCESSES = {
        # Remote Desktop & Screen Sharing
        "anydesk.exe": "AnyDesk Remote Desktop",
        "teamviewer.exe": "TeamViewer Remote Desktop",
        "teamviewer_service.exe": "TeamViewer Service",
        "ultraviewer.exe": "UltraViewer",
        "rustdesk.exe": "RustDesk Remote Access",
        "vncserver.exe": "VNC Server",
        "parsec.exe": "Parsec Screen Share",
        "remotedesktop.exe": "Windows Remote Desktop",
        "mstsc.exe": "Microsoft Terminal Services Client (RDP)",
        # Screen Recorders & Virtual Cameras
        "obs64.exe": "OBS Studio (Screen Capture / Virtual Camera)",
        "obs32.exe": "OBS Studio (Screen Capture)",
        "camtasia.exe": "Camtasia Screen Recorder",
        "bandicam.exe": "Bandicam",
        "sharex.exe": "ShareX Screen Capture",
        # Communication & Voice Cheating
        "discord.exe": "Discord Communication",
        "telegram.exe": "Telegram Messenger",
        "slack.exe": "Slack",
        "whatsapp.exe": "WhatsApp Desktop",
        "zoom.exe": "Zoom Meeting",
        # Virtual Machines / Sandboxes
        "virtualbox.exe": "VirtualBox VM",
        "vmware.exe": "VMware Workstation",
        "vmnat.exe": "VMware NAT Service",
    }

    def __init__(self, prohibited_map: Optional[Dict[str, str]] = None):
        self.prohibited_map = prohibited_map or self.DEFAULT_PROHIBITED_PROCESSES

    def get_connected_display_count(self) -> int:
        """
        Detects the total number of active monitors connected to the system.
        Uses native Windows User32 EnumDisplayMonitors or screen metrics.
        """
        if sys.platform == "win32":
            try:
                monitors = []

                def _enum_callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                    monitors.append(hMonitor)
                    return 1

                enum_proc = ctypes.WINFUNCTYPE(
                    ctypes.c_int,
                    ctypes.c_ulong,
                    ctypes.c_ulong,
                    ctypes.POINTER(ctypes.c_ulong),
                    ctypes.c_double,
                )
                ctypes.windll.user32.EnumDisplayMonitors(
                    0, 0, enum_proc(_enum_callback), 0
                )
                if monitors:
                    return len(monitors)
            except Exception:
                pass

            # Fallback: check VirtualScreen vs PrimaryScreen width
            try:
                sm_cmonitors = 80  # SM_CMONITORS
                count = ctypes.windll.user32.GetSystemMetrics(sm_cmonitors)
                if count > 0:
                    return int(count)
            except Exception:
                pass

        return 1

    def scan_prohibited_processes(self) -> List[Dict[str, Any]]:
        """
        Scans all running processes and flags prohibited cheating/remote tools.
        Returns a list of detected threats with process name, PID, and description.
        """
        detected: List[Dict[str, Any]] = []

        try:
            for proc in psutil.process_iter(["pid", "name", "exe", "create_time"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    if name in self.prohibited_map:
                        detected.append(
                            {
                                "name": name,
                                "pid": proc.info.get("pid"),
                                "description": self.prohibited_map[name],
                                "create_time": proc.info.get("create_time"),
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            print(f"[SystemGuard] Process scan warning: {e}")

        return detected

    def audit_environment(self) -> Dict[str, Any]:
        """
        Executes a complete system audit and returns compliance status.
        """
        display_count = self.get_connected_display_count()
        prohibited_apps = self.scan_prohibited_processes()

        multi_display_violation = display_count > 1 and Config.ENABLE_MULTI_DISPLAY_CHECK
        prohibited_apps_violation = (
            len(prohibited_apps) > 0 and Config.ENABLE_PROCESS_AUDIT
        )

        is_secure = (not multi_display_violation) and (not prohibited_apps_violation)

        return {
            "is_secure": is_secure,
            "display_count": display_count,
            "multi_display_violation": multi_display_violation,
            "prohibited_processes": prohibited_apps,
            "prohibited_apps_count": len(prohibited_apps),
            "prohibited_apps_violation": prohibited_apps_violation,
            "os": sys.platform,
        }
