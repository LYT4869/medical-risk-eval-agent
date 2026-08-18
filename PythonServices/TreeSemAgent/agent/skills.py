from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_AUDIENCES = {"patient", "doctor"}
_SCOPES = {"model_public", "model_technical", "clinical_patient",
           "clinical_professional"}
_AUDIENCE_SCOPES = {
    "patient": {"model_public", "clinical_patient"},
    "doctor": _SCOPES,
}


@dataclass(frozen=True)
class SkillMetadata:
    skill_id: str
    version: str
    display_name: str
    description: str
    audiences: frozenset[str]
    intent_examples: tuple[str, ...]
    required_tools: frozenset[str]
    required_knowledge_scopes: frozenset[str]
    instruction_files: dict[str, Path]
    file_hashes: dict[str, str]


@dataclass(frozen=True)
class SkillActivation:
    skill_id: str
    version: str
    catalog_version: str
    instructions: str
    required_tools: frozenset[str]


class SkillCatalog:
    def __init__(self, root: Path, available_tools: set[str],
                 maximum_file_bytes: int = 16_384):
        self._root = root.resolve()
        self._maximum = maximum_file_bytes
        if not self._root.is_dir() or self._root.is_symlink():
            raise ValueError("trusted skill root must be a real directory")
        self._skills: dict[str, SkillMetadata] = {}
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError("skill root may contain only real directories")
            skill = self._load_metadata(directory.resolve(), available_tools)
            if skill.skill_id in self._skills:
                raise ValueError("duplicate skill id")
            self._skills[skill.skill_id] = skill
        if not self._skills:
            raise ValueError("skill catalog is empty")
        identity = [{"id": item.skill_id, "version": item.version,
                     "files": item.file_hashes}
                    for item in sorted(self._skills.values(), key=lambda value: value.skill_id)]
        self.version = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _safe_file(self, directory: Path, name: str) -> Path:
        if not name or name.startswith(("/", ".")):
            raise ValueError("invalid skill file name")
        path = (directory / name).resolve()
        if directory not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError("skill file escapes trusted directory")
        if path.stat().st_size > self._maximum:
            raise ValueError("skill file is too large")
        return path

    @staticmethod
    def _decode_manifest(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise RuntimeError("PyYAML is required for non-JSON skill manifests") from exc
            value = yaml.safe_load(raw)
        if not isinstance(value, dict):
            raise ValueError("skill manifest must be an object")
        return value

    def _load_metadata(self, directory: Path,
                       available_tools: set[str]) -> SkillMetadata:
        manifest_path = self._safe_file(directory, "skill.yaml")
        manifest = self._decode_manifest(manifest_path.read_text(encoding="utf-8"))
        expected = {"schema_version", "id", "version", "display_name", "description",
                    "audiences", "intent_examples", "required_tools",
                    "required_knowledge_scopes", "instruction_files"}
        if set(manifest) != expected or manifest["schema_version"] != 1:
            raise ValueError("invalid skill manifest schema")
        skill_id = manifest["id"]
        version = manifest["version"]
        if not isinstance(skill_id, str) or not _ID.fullmatch(skill_id):
            raise ValueError("invalid skill id")
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            raise ValueError("invalid skill version")
        audiences = frozenset(manifest["audiences"])
        tools = frozenset(manifest["required_tools"])
        scopes = frozenset(manifest["required_knowledge_scopes"])
        if not audiences or not audiences.issubset(_AUDIENCES):
            raise ValueError("invalid skill audiences")
        if not tools or not tools.issubset(available_tools):
            raise ValueError("skill requires an unknown tool")
        if not scopes.issubset(_SCOPES):
            raise ValueError("skill requires an unknown knowledge scope")
        if any(not scopes.issubset(_AUDIENCE_SCOPES[audience])
               for audience in audiences):
            raise ValueError("skill knowledge scope exceeds an audience capability")
        instructions = manifest["instruction_files"]
        if not isinstance(instructions, dict) or set(instructions) != audiences:
            raise ValueError("skill needs one instruction file per audience")
        instruction_paths = {role: self._safe_file(directory, name)
                             for role, name in instructions.items()}
        examples_path = self._safe_file(directory, "examples.json")
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
        if not isinstance(examples, list) or len(examples) > 64:
            raise ValueError("invalid skill examples")
        intent_examples = manifest["intent_examples"]
        if (not isinstance(intent_examples, list) or not intent_examples or
                any(not isinstance(item, str) or not item.strip() or len(item) > 300
                    for item in intent_examples)):
            raise ValueError("invalid skill intent examples")
        files = {"skill.yaml": manifest_path, "examples.json": examples_path,
                 **{path.name: path for path in instruction_paths.values()}}
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in sorted(files.items())}
        return SkillMetadata(
            skill_id, version, str(manifest["display_name"]),
            str(manifest["description"]), audiences,
            tuple(intent_examples), tools, scopes, instruction_paths, hashes)

    @property
    def count(self) -> int:
        return len(self._skills)

    def summaries(self, role: str) -> list[dict[str, Any]]:
        return [{"id": item.skill_id, "version": item.version,
                 "description": item.description,
                 "required_tools": sorted(item.required_tools),
                 "intent_examples": list(item.intent_examples)}
                for item in sorted(self._skills.values(), key=lambda value: value.skill_id)
                if role in item.audiences]

    def activate(self, skill_id: str, role: str) -> SkillActivation:
        skill = self._skills.get(skill_id)
        if skill is None or role not in skill.audiences:
            raise ValueError("unknown or unavailable skill")
        instructions = skill.instruction_files[role].read_text(encoding="utf-8")
        if not instructions.strip():
            raise ValueError("skill instructions are empty")
        return SkillActivation(skill.skill_id, skill.version, self.version,
                               instructions, skill.required_tools)
