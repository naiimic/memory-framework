from typing import Any, Dict, List, Optional, Union, ClassVar
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseMemory
from memory_architecture.manager.abstract_memory_manager import AbstractMemoryManager
import functools
from sklearn.cluster import DBSCAN
import json
import numpy as np

class Memory(AbstractMemoryManager):
    """Buffer for storing arbitrary memories."""

    llm: BaseLanguageModel

    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_keys: List[str] = None
    prompt_keys: List[str] = None

    workmem: Optional[BaseMemory] = None
    shortmem: Optional[BaseMemory] = None
    longmem: Optional[BaseMemory] = None
    memory_bank: Any = None

    memory_prompts: Optional[Dict[str, str]] = None
    counter: int = 0
    workmem_size_counter: int = 0
    shortmem_size_counter: int = 0

    verbose: bool = False
    name: Any = None
    queried_memories: Any = None
    
    num_cluster_list: ClassVar[list] = []
    num_element_list: ClassVar[list] = []

    def __init__(self, memory_modules, memory_prompts, encoders, memory_bank, executor=None, **data: Any,):
        super().__init__(**data)
        self.memory_keys = list(memory_modules.keys())
        self.prompt_keys = list(memory_prompts.keys())
        self.memory_bank = memory_bank

        default_model_arg = {"encoder": None, "name": self.name}

        model_arg = {
            key: {
                "encoder": encoders.models[encoders.rules[key]],
                "name": self.name,
            }
            if key in encoders.rules else default_model_arg
            for key in self.memory_keys
        }

        # setting up memory modules
        for key, memory_module in memory_modules.items():
            if isinstance(memory_module, functools.partial):
                if key == "shortmem" or key == "longmem":
                    model_arg[key]["memory_type"] = key
                    model_arg[key]["memory_bank"] = memory_bank
                setattr(self, key, memory_module(**model_arg[key]))
            else:
                setattr(self, key, memory_module)

        self.memory_prompts = {
            key: PromptTemplate.from_template("\n".join(val) if isinstance(val, list) else val)
            for key, val in memory_prompts.items() if key in self.prompt_keys
        }

        # Used to keep track of queried memories for data collection
        self.queried_memories = {}

    @property
    def memory_variables(self) -> List[str]:
        return self.memory_keys

    def setup_chain(self, customized_prompt: PromptTemplate = None) -> LLMChain:
        return LLMChain(llm=self.llm, memory=self, prompt=customized_prompt, verbose=self.verbose)

    def fill_memories(self, memories: Dict[str, List]) -> None:
        """Used to sets of memories"""
        for key, value in memories.items():
            if key in self.memory_keys:
                mem_type = getattr(self, key)
                for item in value:
                    # mem_type.add(time=None, content=item)
                    mem_type.add(item)
                    if key == "workmem":
                        self.workmem_size_counter += 1
                    elif key == "shortmem":
                        self.shortmem_size_counter += 1

    def query(self, memory_key: str, text: str, num_memories_queried: int = 1) -> List[str]:
        memory = getattr(self, memory_key)
        responses = memory.query(text=text, num_memories_queried=num_memories_queried)
        return responses

    def threaded_summarize(self, memory_from, memory_to, counter_name=None):
        _mem = self.summarize(memory_from=memory_from, memory_to=memory_to)
        getattr(self, memory_to).add(_mem)
        if counter_name:
            setattr(self, counter_name, getattr(self, counter_name) + 1)

    def update(self, is_last=False) -> None:
        """Called during `slow_forward`, corresponding to conscious reflection"""

        if self.workmem_size_counter >= self.workmem.capacity:
            self.workmem_size_counter = 0
            # Take the oldest memory from workmem and move it to shortmem
            # without summarization (only forgetting when memory enters)
            oldest_memory = getattr(self, 'workmem').items[0]
            getattr(self, 'shortmem').add(oldest_memory)
            # Remove the oldest memory from workmem
            self.workmem.items.pop(0)
            if hasattr(self.workmem, 'items_embeddings') and len(self.workmem.items_embeddings) > 0:
                self.workmem.items_embeddings = self.workmem.items_embeddings[1:]
            setattr(self, 'shortmem_size_counter', getattr(self, 'shortmem').size)

        if self.shortmem_size_counter >= self.shortmem.capacity:
            self.shortmem_size_counter = 0
            # Run clustering, summarize for each cluster, then send to longmem with forgetting
            self.threaded_summarize(memory_from='shortmem', memory_to='longmem')
            self.shortmem.clear()

        return

    def _load_mem(self, memory_key, k=None):
        # if k is an integer, then this is the number of latest memory items to provide
        k = -k if k else None
        return "\n".join(getattr(self, memory_key).items[k:])

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""

        if inputs.get("summarize", False):
            return {}

        memory_vars = {
            "workmem": self._load_mem("workmem"),
            "shortmem": "",
            "longmem": "",
        }

        if inputs.get("update_world_model"):
            pass

        if self.workmem.items:
            query_workmem = self.workmem.items[-1]
            queried_shortmem_workmem = self.shortmem.query(
                query_workmem,
                num_memories_queried=self.shortmem.num_memories_queried,
            )
            queried_longmem_workmem = self.longmem.query(
                query_workmem, num_memories_queried=self.longmem.num_memories_queried
            )
        else:
            queried_shortmem_workmem = []
            queried_longmem_workmem = []

        queried_shortmem = set(queried_shortmem_workmem)
        queried_longmem = set(queried_longmem_workmem)

        if queried_shortmem:
            memory_vars.update({"shortmem": "\n".join(queried_shortmem)})

        if queried_longmem:
            memory_vars.update({"longmem": "\n".join(queried_longmem)})
        return memory_vars

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""

        for key in self.memory_keys:
            if key in inputs:
                self.queried_memories[key] = inputs[key]
        if inputs.get("collect_all_data", False):
            try:
                data = json.loads(outputs["text"])
                data["shortmem"] = inputs["shortmem"]
                data["longmem"] = inputs["longmem"]
                outputs["text"] = json.dumps(data)
            except:
                pass

    def summarize(self, memory_from: str = "shortmem", memory_to: str = "longmem") -> str:
        """Summarize memory_from and add to memory_to"""
        prompt_name = f"{memory_from}_to_{memory_to}"

        chain_input = {
            "name": self.name,
            memory_from: "\n".join(getattr(self, memory_from).items),
            "summarize": True,
        }

        chain = self.setup_chain(customized_prompt=self.memory_prompts[prompt_name])
        return chain.run(chain_input)

    def add(self, content: Optional[Union[dict, str]], key: str = None) -> None:
        self.workmem.add(content)
        self.workmem_size_counter += 1

    def clear(self) -> None:
        for key in self.memory_keys:
            getattr(self, key).clear()

class ChunkedMemory(AbstractMemoryManager):
    """Buffer for storing arbitrary memories."""

    llm: BaseLanguageModel

    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_keys: List[str] = None
    prompt_keys: List[str] = None

    workmem: Optional[BaseMemory] = None
    shortmem: Optional[BaseMemory] = None
    longmem: Optional[BaseMemory] = None
    memory_bank: Any = None

    memory_prompts: Optional[Dict[str, str]] = None
    counter: int = 0
    workmem_size_counter: int = 0
    shortmem_size_counter: int = 0

    verbose: bool = False
    name: Any = None
    queried_memories: Any = None

    num_cluster_average: ClassVar[list] = []
    num_element_list: ClassVar[list] = []

    def __init__(self, memory_modules, memory_prompts, encoders, memory_bank, executor=None, **data: Any,):
        super().__init__(**data)
        self.memory_keys = list(memory_modules.keys())
        self.prompt_keys = list(memory_prompts.keys())
        self.memory_bank = memory_bank

        default_model_arg = {"encoder": None, "name": self.name}

        model_arg = {
            key: {
                "encoder": encoders.models[encoders.rules[key]],
                "name": self.name,
            }
            if key in encoders.rules else default_model_arg
            for key in self.memory_keys
        }

        # setting up memory modules
        for key, memory_module in memory_modules.items():
            if isinstance(memory_module, functools.partial):
                if key == "shortmem" or key == "longmem":
                    model_arg[key]["memory_type"] = key
                    model_arg[key]["memory_bank"] = memory_bank
                setattr(self, key, memory_module(**model_arg[key]))
            else:
                setattr(self, key, memory_module)

        self.memory_prompts = {
            key: PromptTemplate.from_template("\n".join(val) if isinstance(val, list) else val)
            for key, val in memory_prompts.items() if key in self.prompt_keys
        }

        # Used to keep track of queried memories for data collection
        self.queried_memories = {}
        
        # Initialize these in __init__ instead of as class variables
        # self.num_cluster_average = []
        # self.num_element_list = []

    @property
    def memory_variables(self) -> List[str]:
        return self.memory_keys

    def setup_chain(self, customized_prompt: PromptTemplate = None) -> LLMChain:
        return LLMChain(llm=self.llm, memory=self, prompt=customized_prompt, verbose=self.verbose)

    def fill_memories(self, memories: Dict[str, List]) -> None:
        """Used to sets of memories"""
        for key, value in memories.items():
            if key in self.memory_keys:
                mem_type = getattr(self, key)
                for item in value:
                    # mem_type.add(time=None, content=item)
                    mem_type.add(item)
                    if key == "workmem":
                        self.workmem_size_counter += 1
                    elif key == "shortmem":
                        self.shortmem_size_counter += 1

    def query(self, memory_key: str, text: str, num_memories_queried: int = 1) -> List[str]:
        memory = getattr(self, memory_key)
        responses = memory.query(text=text, num_memories_queried=num_memories_queried)
        return responses

    def threaded_summarize(self, memory_from, memory_to, counter_name=None):
        _mem = self.summarize(memory_from=memory_from, memory_to=memory_to)
        for m in _mem:
            getattr(self, memory_to).add(m)

    def update(self, is_last=False) -> None:
        """Called during `slow_forward`, corresponding to conscious reflection"""

        if self.workmem_size_counter >= self.workmem.capacity:
            self.workmem_size_counter = 0
            # Take the oldest memory from workmem and move it to shortmem
            # without summarization (only forgetting when memory enters)
            oldest_memory = getattr(self, 'workmem').items[0]
            getattr(self, 'shortmem').add(oldest_memory)
            # Remove the oldest memory from workmem
            self.workmem.items.pop(0)
            if hasattr(self.workmem, 'items_embeddings') and len(self.workmem.items_embeddings) > 0:
                self.workmem.items_embeddings = self.workmem.items_embeddings[1:]
            setattr(self, 'shortmem_size_counter', getattr(self, 'shortmem').size)

        if self.shortmem_size_counter >= self.shortmem.capacity:
            self.shortmem_size_counter = 0
            # Run clustering, summarize for each cluster, then send to longmem with forgetting
            self.threaded_summarize(memory_from='shortmem', memory_to='longmem')
            self.shortmem.clear()

        return

    def _load_mem(self, memory_key, k=None):
        # if k is an integer, then this is the number of latest memory items to provide
        k = -k if k else None
        return "\n".join(getattr(self, memory_key).items[k:])

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""

        if inputs.get("summarize", False):
            return {}

        memory_vars = {
            "workmem": self._load_mem("workmem"),
            "shortmem": "",
            "longmem": "",
        }

        if inputs.get("update_world_model"):
            pass

        if self.workmem.items:
            query_workmem = self.workmem.items[-1]
            queried_shortmem_workmem = self.shortmem.query(
                query_workmem,
                num_memories_queried=self.shortmem.num_memories_queried,
            )
            queried_longmem_workmem = self.longmem.query(
                query_workmem, num_memories_queried=self.longmem.num_memories_queried
            )
        else:
            queried_shortmem_workmem = []
            queried_longmem_workmem = []

        queried_shortmem = set(queried_shortmem_workmem)
        queried_longmem = set(queried_longmem_workmem)

        if queried_shortmem:
            memory_vars.update({"shortmem": "\n".join(queried_shortmem)})

        if queried_longmem:
            memory_vars.update({"longmem": "\n".join(queried_longmem)})
        return memory_vars

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""

        for key in self.memory_keys:
            if key in inputs:
                self.queried_memories[key] = inputs[key]
        if inputs.get("collect_all_data", False):
            try:
                data = json.loads(outputs["text"])
                data["shortmem"] = inputs["shortmem"]
                data["longmem"] = inputs["longmem"]
                outputs["text"] = json.dumps(data)
            except:
                pass

    def summarize(self, memory_from: str = "shortmem", memory_to: str = "longmem") -> str:
        """Summarize memory_from and add to memory_to"""

        dbscan = DBSCAN(eps=0.55, min_samples=1)
        labels = dbscan.fit_predict(getattr(self, memory_from).items_embeddings)
        
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            
            clusters_dict[label].append(idx)
        
        clusters_list = list(clusters_dict.values())
        
        # record the number of clusters and elements
        if memory_from == "shortmem":
        # print(f'len_cluster_list={len(clusters_list)}, mean_elements={clusters_list}')
            self.num_cluster_average.append(len(clusters_list))
        
        prompt_name = f"{memory_from}_to_{memory_to}"
        all_items = getattr(self, memory_from).items

        _mem = []
        
        for cluster in clusters_list:

            if len(cluster) == 1:
                _mem.append(all_items[cluster[0]])
            else:
                chain_input = {
                    "name": self.name,
                    memory_from: "\n".join([all_items[idx] for idx in cluster]),
                    "summarize": True,
                }

                chain = self.setup_chain(customized_prompt=self.memory_prompts[prompt_name])
                _mem.append(chain.run(chain_input))
        
        return _mem

    def add(self, content: Optional[Union[dict, str]], key: str = None) -> None:
        self.workmem.add(content)
        self.workmem_size_counter += 1

    def clear(self) -> None:
        for key in self.memory_keys:
            getattr(self, key).clear()