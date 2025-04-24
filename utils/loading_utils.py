from utils.encoder_utils import OpenAIEncoder, EncoderCollection
from memory_architecture.manager.default_manager import Memory, ChunkedMemory
from memory_architecture.module.default_modules import MemoryStore, EmbeddingMemory
from langchain.chat_models import ChatOpenAI
from omegaconf import OmegaConf
from utils.analysis_utils import save_and_log_state
from tqdm import tqdm
import random
from settings import PINECONE_APIKEY, PINECONE_ENVIRONMENT, PINECONE_INDEX

if PINECONE_APIKEY != None:
    import pinecone
    import time

def initialize_memory_architecture_chunk(MEMORY_CONFIG, workmem_pairs = None, recentmem_pairs = None, longmem_pairs = None, openai_encoder = None):
    list_config = OmegaConf.create(["workmem", "recentmem", "longmem"])
    encoder_collection = EncoderCollection(models={"openai": openai_encoder}, rules={"recentmem": "openai", "longmem": "openai"}, validation_config=list_config)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    workmem = MemoryStore(capacity=MEMORY_CONFIG["workmem_capacity"], name="workmem")

    if PINECONE_INDEX is not None:
        pinecone.init(api_key=PINECONE_APIKEY, environment=PINECONE_ENVIRONMENT)
        time.sleep(1)
        memory_bank = pinecone.Index(index_name=PINECONE_INDEX)
        time.sleep(1)
        
        # Simple test
        # Change to specific namespace in the future
        memory_bank.upsert(
            [
                ("A", [random.random() for _ in range(1536)]),
                ("B", [random.random() for _ in range(1536)]),
            ],
            namespace="general-space",
        )
        memory_bank.delete(delete_all=True, namespace="general-space")
    else:
        memory_bank = []

    recentmem = EmbeddingMemory(capacity=MEMORY_CONFIG["recentmem_capacity"], name="recentmem", encoder=openai_encoder, num_memories_queried=MEMORY_CONFIG["recentmem_num_memories_queried"],forgetting_algorithm=MEMORY_CONFIG["recentmem_forgetting_algorithm"],forgetting_threshold=MEMORY_CONFIG["recentmem_forgetting_threshold"],memory_type = 'recentmem', memory_bank = memory_bank)
    longmem = EmbeddingMemory(capacity=MEMORY_CONFIG["longmem_capacity"], name="longmem", encoder=openai_encoder, num_memories_queried=MEMORY_CONFIG["longmem_num_memories_queried"], forgetting_algorithm=MEMORY_CONFIG["longmem_forgetting_algorithm"], forgetting_threshold=MEMORY_CONFIG["longmem_forgetting_threshold"], memory_type = 'longmem', memory_bank = memory_bank)

    if workmem_pairs:
        workmem.memory_bank = [memory_content for memory_content, _ in workmem_pairs[-workmem.capacity:]]

    if recentmem_pairs:
        recentmem.memory_bank = [memory_content for memory_content, _ in recentmem_pairs[-recentmem.capacity:]]
        recentmem.embeddings = [embedding for _, embedding in recentmem_pairs[-recentmem.capacity:]]

    if longmem_pairs:
        longmem.memory_bank = [memory_content for memory_content, _ in longmem_pairs[-longmem.capacity:]]
        longmem.embeddings = [embedding for _, embedding in longmem_pairs[-longmem.capacity:]]

    memory_prompts = {
        "workmem_to_recentmem": "You are {name}. Provide a brief summary of the following working memories: {workmem} Do not include times in the summary. Summarize the most stricking. Do not change the use of pronouns in the summary. (Less than 25 words)",
        "recentmem_to_longmem": "You are {name}. Extract from the following memories the most important content: {recentmem}. Do not include times in the summary. Do not change the use of pronouns in the summary. (Less than 25 words)"
    }

    memory_modules = {
        "workmem": workmem, 
        "recentmem":recentmem,
        "longmem": longmem
    }

    memory_class = globals()[MEMORY_CONFIG["memory_class"]]
    manager = memory_class(name = MEMORY_CONFIG["name"],llm=llm, memory_modules=memory_modules, memory_prompts=memory_prompts, encoders=encoder_collection, memory_bank = [])

    return manager

def simulate_memory_stream_chunk(events, MEMORY_CONFIG):

    openai_encoder = OpenAIEncoder(model_name="text-embedding-ada-002")
    memory_manager = initialize_memory_architecture_chunk(MEMORY_CONFIG, openai_encoder = openai_encoder)
    m_start = 0
    workmem_pairs = []

    for m, _mem in tqdm(enumerate(events[m_start:], start=m_start), total=len(events[m_start:])):

        memory_manager.add(content=_mem)
        workmem_pairs.append((_mem, openai_encoder([_mem])))
        memory_manager.update()

        save_and_log_state(m, len(events[m_start:]), workmem_pairs, memory_manager)

    return memory_manager, workmem_pairs