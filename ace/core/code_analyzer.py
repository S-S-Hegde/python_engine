"""
ace/core/code_analyzer.py - Keystroke Dynamics Biometrics & AST Code Plagiarism Analyzer.
Calculates typing cadence (WPM, inter-arrival time standard deviation), detects paste bursts,
and parses Python/JS code AST structures to identify boilerplate AI generation.
"""

import ast
import time
import math
import collections
from typing import Dict, List, Any, Optional, Tuple


class CodeIntegrityAnalyzer:
    """
    Analyzes live keystroke telemetry streams and completed source code
    to detect abnormal paste injections, auto-typers, and AI-templated code.
    """

    def __init__(self):
        # Keystroke history buffer: [(timestamp_ms, key_type, char_count, code_len)]
        self._keystrokes: List[Dict[str, Any]] = []
        self._paste_events: List[Dict[str, Any]] = []
        self._tab_switch_events: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None

    def record_keystroke(
        self,
        event_type: str,
        code_length: int,
        chars_added: int = 1,
        key: str = "",
        timestamp_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a real-time keystroke event and calculates instant typing velocity.
        """
        now = (timestamp_ms / 1000.0) if timestamp_ms else time.time()
        if self._start_time is None:
            self._start_time = now

        entry = {
            "time": now,
            "type": event_type,  # 'key', 'paste', 'backspace', 'cut'
            "length": code_length,
            "added": chars_added,
            "key": key,
        }
        self._keystrokes.append(entry)

        # Flag instant bulk injection (e.g. > 15 characters added in a single event without typing)
        is_paste_burst = (event_type == "paste") or (chars_added > 15 and event_type != "key")
        if is_paste_burst:
            self._paste_events.append(
                {
                    "time": now,
                    "chars_added": chars_added,
                    "code_length": code_length,
                }
            )

        return {
            "is_paste_burst": is_paste_burst,
            "total_keystrokes": len(self._keystrokes),
            "paste_strikes": len(self._paste_events),
        }

    def record_tab_switch(self, event_type: str, details: str = "") -> int:
        """Records a browser tab blur / visibilitychange event."""
        now = time.time()
        self._tab_switch_events.append(
            {
                "time": now,
                "event": event_type,  # 'blur', 'hidden', 'fullscreen_exit'
                "details": details,
            }
        )
        return len(self._tab_switch_events)

    def compute_keystroke_biometrics(self) -> Dict[str, Any]:
        """
        Computes overall typing biometrics: WPM, average cadence, typing fluidity,
        and paste burst anomalies.
        """
        if len(self._keystrokes) < 5:
            return {
                "wpm": 0.0,
                "total_keystrokes": len(self._keystrokes),
                "paste_events_count": len(self._paste_events),
                "tab_switch_count": len(self._tab_switch_events),
                "typing_fluidity_score": 100.0,
                "suspicious_speed": False,
            }

        # Calculate time span
        t_first = self._keystrokes[0]["time"]
        t_last = self._keystrokes[-1]["time"]
        total_seconds = max(1.0, t_last - t_first)
        minutes = total_seconds / 60.0

        # Calculate standard character count and words (5 chars = 1 word)
        total_chars_typed = sum(k["added"] for k in self._keystrokes if k["type"] in ("key", "backspace"))
        wpm = (total_chars_typed / 5.0) / minutes if minutes > 0 else 0.0

        # Inter-key latency standard deviation
        intervals = []
        for i in range(1, len(self._keystrokes)):
            dt = self._keystrokes[i]["time"] - self._keystrokes[i - 1]["time"]
            if 0 < dt < 3.0:  # ignore long pauses
                intervals.append(dt)

        if len(intervals) > 2:
            mean_int = sum(intervals) / len(intervals)
            variance = sum((x - mean_int) ** 2 for x in intervals) / len(intervals)
            std_dev = math.sqrt(variance)
            # Fluidity: natural typing has moderate jitter (std_dev > 0.04s)
            # Robotic script/auto-typer has near 0 jitter (std_dev < 0.01s)
            is_robotic_typer = std_dev < 0.015 and len(intervals) > 20
        else:
            std_dev = 0.1
            is_robotic_typer = False

        suspicious_speed = wpm > 160.0 or is_robotic_typer or len(self._paste_events) > 0

        fluidity_score = 100.0
        if len(self._paste_events) > 0:
            fluidity_score -= min(60.0, len(self._paste_events) * 20.0)
        if is_robotic_typer:
            fluidity_score -= 40.0
        if wpm > 160:
            fluidity_score -= 30.0

        return {
            "wpm": round(wpm, 1),
            "total_keystrokes": len(self._keystrokes),
            "paste_events_count": len(self._paste_events),
            "tab_switch_count": len(self._tab_switch_events),
            "inter_key_jitter": round(std_dev, 4),
            "is_robotic_typer": is_robotic_typer,
            "suspicious_speed": suspicious_speed,
            "typing_fluidity_score": max(0.0, round(fluidity_score, 1)),
        }

    def analyze_python_ast(self, source_code: str) -> Dict[str, Any]:
        """
        Parses submitted Python code using the Python AST engine to inspect
        syntactic complexity, node distributions, and detect boilerplate structure.
        """
        if not source_code or not source_code.strip():
            return {"valid_syntax": False, "nodes_count": 0, "complexity_score": 0}

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return {
                "valid_syntax": False,
                "syntax_error": f"Line {e.lineno}: {e.msg}",
                "nodes_count": 0,
                "complexity_score": 0,
            }

        node_types = collections.Counter(type(node).__name__ for node in ast.walk(tree))
        total_nodes = sum(node_types.values())

        # Measure structural complexity (loops, conditionals, function definitions)
        complexity_nodes = (
            node_types.get("If", 0)
            + node_types.get("For", 0)
            + node_types.get("While", 0)
            + node_types.get("FunctionDef", 0)
            + node_types.get("AsyncFunctionDef", 0)
            + node_types.get("Try", 0)
            + node_types.get("ListComp", 0)
        )
        cyclomatic_approx = max(1, complexity_nodes + 1)

        # Detect verbose LLM comment headers / docstrings
        has_docstrings = node_types.get("Expr", 0) > 0 and '"""' in source_code
        comment_lines = [l for l in source_code.splitlines() if l.strip().startswith("#")]

        return {
            "valid_syntax": True,
            "total_ast_nodes": total_nodes,
            "functions_count": node_types.get("FunctionDef", 0),
            "loops_count": node_types.get("For", 0) + node_types.get("While", 0),
            "branching_count": node_types.get("If", 0),
            "cyclomatic_complexity": cyclomatic_approx,
            "comment_lines_count": len(comment_lines),
            "has_docstrings": has_docstrings,
            "node_distribution": dict(node_types.most_common(8)),
        }

    def evaluate_submission(self, source_code: str) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation combining keystroke biometrics and code structure.
        """
        biometrics = self.compute_keystroke_biometrics()
        ast_report = self.analyze_python_ast(source_code)

        # Integrity Score (0 - 100)
        integrity = biometrics["typing_fluidity_score"]
        if biometrics["tab_switch_count"] > 0:
            integrity -= min(40.0, biometrics["tab_switch_count"] * 10.0)

        integrity = max(0.0, min(100.0, integrity))
        is_clean = integrity >= 70.0 and biometrics["paste_events_count"] == 0

        return {
            "integrity_score": round(integrity, 1),
            "is_clean": is_clean,
            "biometrics": biometrics,
            "ast_analysis": ast_report,
            "summary": "Authentic candidate keystrokes"
            if is_clean
            else f"Integrity warning: {biometrics['paste_events_count']} pastes, {biometrics['tab_switch_count']} tab switches.",
        }
