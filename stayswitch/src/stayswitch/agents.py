"""Harbor agents for StaySwitch runs. Use with ``--agent stayswitch.agents:StaySwitchTerminus``.

``StaySwitchTerminus`` is terminus-2 with two changes:

* Its LLM client sends the trajectory's session id (``X-Session-ID``). Stock
  terminus-2 generates a session id but never hands it to its LLM client.
* Before setup, the task container's apt sources are pointed at a nearby
  mirror (``STAYSWITCH_APT_MIRROR``, default Tsinghua; empty disables). From
  this network ports.ubuntu.com takes ~90 s for the tmux install that
  terminus-2 caps at 120 s, which fails trials before the agent starts.
"""

from __future__ import annotations

import os
import shlex

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession

DEFAULT_APT_MIRROR = "http://mirrors.tuna.tsinghua.edu.cn"

# Distribution hosts rewritten to the mirror; the path layout is identical on Tsinghua.
_APT_HOSTS = ("http://ports.ubuntu.com", "http://archive.ubuntu.com", "http://security.ubuntu.com", "http://deb.debian.org")

# Headroom for the tmux/asciinema install when a mirror is unavailable (stock: 120 s per command, 240 s total).
TmuxSession._TOOL_INSTALL_TIMEOUT_SEC = 300
TmuxSession._TOOL_INSTALL_BUDGET_SEC = 600


def apt_mirror_command(mirror: str) -> str:
    seds = " ".join(f"-e {shlex.quote(f's#{host}#{mirror}#g')}" for host in _APT_HOSTS)
    files = "/etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list"
    return f"for f in {files}; do [ -f \"$f\" ] && sed -i {seds} \"$f\"; done; true"


class StaySwitchClaudeCode(ClaudeCode):
    """Claude Code with the task container's apt sources pointed at a nearby mirror before install.

    Harbor installs Claude Code by apt-installing curl/nodejs/npm/procps, which takes over
    ten minutes from ports.ubuntu.com here, then running Anthropic's bootstrap script.
    Use with ``--agent stayswitch.agents:StaySwitchClaudeCode``.
    """

    @staticmethod
    def name() -> str:
        return "stayswitch-claude-code"

    async def install(self, environment) -> None:
        mirror = os.environ.get("STAYSWITCH_APT_MIRROR", DEFAULT_APT_MIRROR)
        if mirror:
            await environment.exec(command=apt_mirror_command(mirror), user="root", timeout_sec=30)
        await super().install(environment)


class StaySwitchTerminus(Terminus2):
    @staticmethod
    def name() -> str:
        return "stayswitch-terminus"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Terminus2 builds its LLM from options.session_id (None by default) before creating self._session_id.
        if getattr(self._llm, "_session_id", "unset") is None:
            self._llm._session_id = self._session_id

    async def setup(self, environment) -> None:
        mirror = os.environ.get("STAYSWITCH_APT_MIRROR", DEFAULT_APT_MIRROR)
        if mirror:
            await environment.exec(command=apt_mirror_command(mirror), user="root", timeout_sec=30)
        await super().setup(environment)
