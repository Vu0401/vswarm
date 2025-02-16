from typing import Optional, Dict
from pydantic import BaseModel, ConfigDict

from swarm.memory import EntityMemory, LongTermMemory, ShortTermMemory


class ContextualMemory(BaseModel):
    """
    A class that synthesizes context from different memory types.
    Handles optional memory components and provides context building functionality.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    short_term: Optional[ShortTermMemory] = None
    long_term: Optional[LongTermMemory] = None
    entity: Optional[EntityMemory] = None

    def synthesize(self, query: str) -> Dict[str, list]:
        """
        Synthesize context from all available memory types for a given query.

        Args:
            query: The query to search for relevant context

        Returns:
            Dict containing consolidated context from all memory types
        """
        context = []

        # Get long-term memory context
        if self.long_term:
            ltm_context = self._get_ltm_context(query)
            if ltm_context:
                context.extend(ltm_context)

        # Get short-term memory context
        if self.short_term:
            stm_context = self._get_stm_context(query)
            if stm_context:
                context.extend(stm_context)

        # Get entity memory context
        if self.entity:
            entity_context = self._get_entity_context(query)
            if entity_context:
                context.extend(entity_context)

        return {"context": context}

    def _get_ltm_context(self, task: str) -> list:
        """Get relevant context from long-term memory"""
        context = []
        results = self.long_term.search(task, latest_n=2)

        if results:
            for result in results:
                if "suggestions" in result["metadata"]:
                    suggestions = result["metadata"]["suggestions"]
                    context.extend(
                        [
                            f"Previous task suggestion: {suggestion}"
                            for suggestion in suggestions
                        ]
                    )

        return context

    def _get_stm_context(self, query: str) -> list:
        """Get relevant context from short-term memory"""
        context = []
        results = self.short_term.search(query)

        if results:
            context.extend(
                [f"Recent interaction: {result['documents']}" for result in results]
            )

        return context

    def _get_entity_context(self, query: str) -> list:
        """Get relevant context from entity memory"""
        context = []
        results = self.entity.search(query)

        if results:
            for result in results:
                entity_type = result["metadata"].get("type", "Unknown")
                entity_name = result["metadata"].get("name", "Unknown")
                description = result["documents"]

                context.append(f"Known {entity_type} '{entity_name}': {description}")

        return context
