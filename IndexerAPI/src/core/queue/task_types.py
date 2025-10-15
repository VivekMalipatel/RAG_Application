from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

class TaskType(Enum):
    FILE = "file"
    URL = "url"
    TEXT = "text"
    UNSTRUCTURED_PAGE = "unstructured_page"
    STRUCTURED_CHUNK = "structured_chunk"
    DIRECT_CHUNK = "direct_chunk"

@dataclass
class TaskMessage:
    task_id: str
    task_type: TaskType
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "task_type": self.task_type.value, "payload": self.payload}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMessage":
        return cls(task_id=data["task_id"], task_type=TaskType(data["task_type"]), payload=data["payload"])
