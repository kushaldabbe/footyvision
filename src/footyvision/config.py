from dataclasses import dataclass

@dataclass
class Config:
    some_setting: str = "default_value"