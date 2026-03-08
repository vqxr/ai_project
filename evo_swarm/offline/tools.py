from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional

from evo_swarm.offline.config import OfflineSwarmConfig
from evo_swarm.offline.knowledge.store import KnowledgeStore


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    output: str


class ToolRunner:
    def __init__(self, *, config: OfflineSwarmConfig, store: KnowledgeStore, workspace_root: str):
        self.config = config
        self.store = store
        self.workspace_root = os.path.abspath(workspace_root)

    def run(self, call: ToolCall) -> ToolResult:
        try:
            if call.name == "search_papers":
                q = str(call.args.get("query", ""))
                limit = int(call.args.get("limit", 8))
                hits = self.store.search(q, limit=limit)
                lines = []
                for h in hits:
                    snippet = h.text[:400].replace("\n", " ").strip()
                    lines.append(f"- {h.doc_path} @ {h.offset} score={h.score:.3f}\n  {snippet}")
                return ToolResult(name=call.name, ok=True, output="\n".join(lines) if lines else "(no hits)")

            if call.name == "read_file":
                path = os.path.abspath(str(call.args["path"]))
                self._assert_within_workspace(path)
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return ToolResult(name=call.name, ok=True, output=f.read())

            if call.name == "write_file":
                path = os.path.abspath(str(call.args["path"]))
                self._assert_within_workspace(path)
                content = str(call.args.get("content", ""))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult(name=call.name, ok=True, output=f"WROTE {path} ({len(content)} bytes)")

            if call.name == "run_cmd":
                cmd = str(call.args["cmd"])
                timeout_s = int(call.args.get("timeout_s", 60))
                cwd = os.path.abspath(str(call.args.get("cwd", self.workspace_root)))
                self._assert_within_workspace(cwd)
                self._assert_allowed_command(cmd)
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    shell=True,
                    text=True,
                    capture_output=True,
                    timeout=timeout_s,
                )
                out = (proc.stdout or "") + (proc.stderr or "")
                ok = proc.returncode == 0
                return ToolResult(name=call.name, ok=ok, output=out.strip() or f"(exit {proc.returncode})")

            return ToolResult(name=call.name, ok=False, output=f"Unknown tool: {call.name}")
        except Exception as e:
            return ToolResult(name=call.name, ok=False, output=str(e))

    def _assert_within_workspace(self, path: str) -> None:
        root = self.workspace_root.rstrip(os.sep) + os.sep
        p = os.path.abspath(path)
        if not (p == self.workspace_root or p.startswith(root)):
            raise ValueError(f"path escapes workspace: {p}")

    def _assert_allowed_command(self, cmd: str) -> None:
        # Validate by inspecting the first token after shell parsing.
        tokens = shlex.split(cmd, posix=True)
        if not tokens:
            raise ValueError("empty cmd")
        head = tokens[0]
        for allowed in self.config.allowed_cmd_prefixes:
            if head == allowed or cmd.startswith(allowed + " "):
                return
        raise ValueError(f"cmd not allowed: {head}")

