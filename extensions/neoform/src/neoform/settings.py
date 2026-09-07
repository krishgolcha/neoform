from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    adapter: str
    host: str
    port: int
    web_origin: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.getenv("NEOFORM_DATA_DIR", ".neoform")).resolve(),
            adapter=os.getenv("NEOFORM_ADAPTER", "tinker"),
            host=os.getenv("NEOFORM_HOST", "127.0.0.1"),
            port=int(os.getenv("NEOFORM_PORT", "8787")),
            web_origin=os.getenv("NEOFORM_WEB_ORIGIN", "http://localhost:3000"),
        )

