import logging
import random
from typing import Any, List
from threading import Lock, Event
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Any
from utils.encoder_utils import EncoderManager
from utils.utils import get_colored_text
from utils.memory_utils import retrieve

logger = logging.getLogger(__name__)

# WM
class MemoryStore:
    """Buffer for storing arbitrary memories."""

    def __init__(
        self,
        buffer: List[Dict[str, Union[str, Dict[str, Any]]]] = None,
        encoder: Any = None,
        capacity: int = 1_000,
        metadata_types: List[str] = None,
        name=None,
    ):
        logger.debug(f"Initializing MemoryStore {name}...")
        self.name = name
        self.capacity = capacity
        self.metadata_type = metadata_types
        self.memory_bank = buffer or []

        assert encoder is None, "encoder must be None"
        assert (len(self.memory_bank) <= self.capacity), "Memory count must not exceed capacity"

    @property
    def items(self) -> List[str]:
        return self.memory_bank

    @property
    def size(self) -> int:
        return len(self.memory_bank)

    def add(self, memory: str) -> None:
        """Add a memory to the buffer."""

        # Add memory to bank or remove the oldest one if capacity is reached
        if len(self.memory_bank) >= self.capacity:
            self.memory_bank.pop(0)
        self.memory_bank.append(memory)

    def clear(self) -> None:
        """Clear all memories."""
        self.memory_bank = []

# RM and LM
class EmbeddingMemory:
    """Buffer for storing memories with embeddings."""

    def __init__(
        self,
        buffer: List[str] = None,
        encoder: Any = None,
        capacity: int = 1_000,
        num_memories_queried: int = 2,
        forgetting_algorithm=True,
        forgetting_threshold: float = 0.9,
        name=None,
        memory_type: Any = None,
        memory_bank: Any = None,
    ):
        logger.debug(f"Initializing EmbeddingMemory {name}...")

        # Basic properties
        self.agent_name = name
        self.encoder = encoder
        self.capacity = capacity
        self.num_memories_queried = num_memories_queried
        self.forgetting_algorithm = forgetting_algorithm
        self.forgetting_threshold = forgetting_threshold

        # Memory info
        self.memory_type = memory_type

        # Classic local memory
        self.memory_bank = []
        self.embeddings = []

        # Threading utilities
        self.lock = Lock()
        self.query_embeddings = []
        self.query_event = Event()

    @property
    def items(self) -> List[str]:
        return self.memory_bank
        
    @property
    def items_embeddings(self) -> List[str]:
        return self.embeddings

    @property
    def size(self) -> int:
        return len(self.memory_bank)

    # Forgetting algorithm
    def _update_memory_bank(self, new_embedding, new_memory_content):
        """Update memory bank by removing similar memories based on embeddings."""

        similarities = cosine_similarity(new_embedding.reshape(1, -1), self.embeddings)

        # Find indices of similar memories (similarities above or equal to the threshold)
        similar_indices = [
            i
            for i, value in enumerate(similarities[0])
            if value >= self.forgetting_threshold
        ]

        # Remove the similar memories and embeddings based on their indices
        for index in sorted(similar_indices, reverse=True):
            del self.memory_bank[index]
            del self.embeddings[index]

    def add(self, memory: str) -> None:
        """Add a memory to the memory bank."""

        if len(self.memory_bank) == self.capacity:
            with self.lock:
                self.memory_bank.pop(0)
                self.embeddings.pop(0)

        logger.debug(f"Queueing memory to {self.agent_name} module")
        self.encoder.queue_document((self, memory))
        logger.debug(f"Adding memory: {get_colored_text(text=memory, color='blue')}")

    def fill_encoder_memories(self, embedding, memory_content):
        """Add memories that come from encoder."""

        if self.forgetting_algorithm and self.memory_bank:
            self._update_memory_bank(embedding, memory_content)

        self.memory_bank.append(memory_content)
        self.embeddings.append(embedding)

    def query(self, text: str, num_memories_queried: int = 1, timeout=5.0) -> List[str]:
        """Retrieve memories based on query content."""

        if not isinstance(self.encoder, EncoderManager):
            self.encoder.queue_query((self, text))
            results = self.retrieve(num_memories_queried)
            logger.debug(
                f"Query result: {get_colored_text(text=str(results), color='blue')}"
            )
            return results

        self._async_query(text)

        if not self.query_event.wait(timeout):
            logger.warning(
                f"Query timed out after {timeout} seconds. Returning empty list."
            )
            return []

        results = self.retrieve(num_memories_queried)
        logger.debug(
            f"Query result: {get_colored_text(text=str(results), color='blue')}"
        )

        return results

    def _async_query(self, text: str):
        self.query_event.clear()
        self.encoder.queue_query((self, text))

    def retrieve(self, num_memories_queried: int) -> List[str]:
        """Retrieve memories based on current query embeddings."""

        results = [
            memory
            for embed in self.query_embeddings
            for memory in retrieve(
                embed, self.embeddings, self.memory_bank, num_memories_queried
            )
        ]

        with self.lock:
            self.query_embeddings = []

        return results

    def clear(self) -> None:
        """Clear all memories and embeddings."""
        self.memory_bank = []
        self.embeddings = []
        self.query_embeddings = []
