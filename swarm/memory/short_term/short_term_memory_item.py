from typing import Any, Dict, Optional


class ShortTermMemoryItem:
    def __init__(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.metadata = metadata if metadata is not None else {}
