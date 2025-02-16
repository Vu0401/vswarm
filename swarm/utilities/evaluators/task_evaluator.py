from typing import List

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="The name of the entity.")
    type: str = Field(description="The type of the entity.")
    description: str = Field(description="Description of the entity.")
    relationships: List[str] = Field(description="Relationships of the entity.")


class TaskEvaluation(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve future similar tasks."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance, all taking into account the task description, expected output, and the result of the task."
    )
    entities: List[Entity] = Field(
        description="Entities extracted from the task output."
    )
