from typing import Any, Dict, Optional, Union


class LongTermMemoryItem:
    def __init__(
        self,
        agent: str,
        task: str,
        datetime: str,
        expected_output: Optional[str] = None,
        quality: Optional[Union[int, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task = task
        self.agent = agent
        self.quality = quality
        self.datetime = datetime
        self.expected_output = expected_output
        self.metadata = metadata if metadata is not None else {}
