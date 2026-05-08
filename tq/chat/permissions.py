from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Optional


class PermissionAction:
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRule:
    pattern: str
    action: str


@dataclass
class PermissionConfig:
    rules: dict[str, list[PermissionRule]] = field(default_factory=dict)

    def get_action(self, tool: str, input_value: str = "", workdir: str = "") -> str:
        rules = self.rules.get(tool, [])
        if not rules:
            default_rules = self.rules.get("*", [])
            rules = default_rules

        result = PermissionAction.ALLOW
        for rule in rules:
            if self._matches(rule.pattern, input_value, workdir):
                result = rule.action

        if tool in ("edit", "write", "apply_patch") and input_value:
            abs_input = os.path.abspath(input_value)
            abs_workdir = os.path.abspath(workdir) if workdir else os.getcwd()
            if not abs_input.startswith(abs_workdir):
                ext_rules = self.rules.get("external_directory", [])
                if ext_rules:
                    for rule in ext_rules:
                        if self._matches(rule.pattern, input_value, workdir):
                            if rule.action == PermissionAction.ALLOW:
                                return result
                    return PermissionAction.ASK

        return result

    def _matches(self, pattern: str, value: str, workdir: str = "") -> bool:
        if pattern == "*":
            return True

        expanded = pattern
        if expanded.startswith("~"):
            expanded = os.path.expanduser(expanded)
        if "$HOME" in expanded:
            expanded = expanded.replace("$HOME", os.path.expanduser("~"))

        return fnmatch.fnmatch(value, expanded)

    @classmethod
    def from_dict(cls, data: dict) -> PermissionConfig:
        rules: dict[str, list[PermissionRule]] = {}
        for tool, val in data.items():
            if isinstance(val, str):
                rules[tool] = [PermissionRule(pattern="*", action=val)]
            elif isinstance(val, dict):
                tool_rules = []
                for pattern, action in val.items():
                    tool_rules.append(PermissionRule(pattern=pattern, action=action))
                rules[tool] = tool_rules
        return cls(rules=rules)

    @classmethod
    def defaults(cls) -> PermissionConfig:
        return cls.from_dict({
            "read": {"*": PermissionAction.ALLOW, "*.env": PermissionAction.DENY, "*.env.*": PermissionAction.DENY},
            "edit": {"*": PermissionAction.ALLOW},
            "write": {"*": PermissionAction.ALLOW},
            "bash": {"*": PermissionAction.ALLOW},
            "glob": {"*": PermissionAction.ALLOW},
            "grep": {"*": PermissionAction.ALLOW},
            "apply_patch": {"*": PermissionAction.ALLOW},
            "webfetch": {"*": PermissionAction.ALLOW},
            "websearch": {"*": PermissionAction.ALLOW},
            "question": {"*": PermissionAction.ALLOW},
            "todowrite": {"*": PermissionAction.ALLOW},
            "external_directory": {"*": PermissionAction.ASK},
            "doom_loop": {"*": PermissionAction.ASK},
        })


def ask_permission(tool: str, input_value: str) -> str:
    print(f"\n  \033[33m{tool}\033[0m wants to: {input_value}")
    print("  \033[2m[o]nce / [a]lways / [r]eject\033[0m")
    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return PermissionAction.DENY

    if choice in ("a", "always"):
        return PermissionAction.ALLOW
    if choice in ("o", "once", ""):
        return PermissionAction.ALLOW
    return PermissionAction.DENY
