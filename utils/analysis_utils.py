import os
import yaml
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

CONTENT_LOG_FREQUENCY = 500  # Change as desired; logs actual content every nth iteration.

def serialize_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def save_and_log_state(iteration, total_iterations, workmem_pairs, memory_manager):
    if iteration % CONTENT_LOG_FREQUENCY == 0 or iteration == total_iterations - 1:  # Store actual content on every nth iteration.
        shortmem_pairs = [(mem, serialize_to_json(emb)) for mem, emb in zip(memory_manager.shortmem.items, memory_manager.shortmem.items_embeddings)]
        longmem_pairs = [(mem, serialize_to_json(emb)) for mem, emb in zip(memory_manager.longmem.items, memory_manager.longmem.items_embeddings)]
    else:
        # Only store memory sizes if not storing actual content.
        shortmem_pairs = memory_manager.shortmem.size
        longmem_pairs = memory_manager.longmem.size

    log_entry = {
        "action": "save_state",
        "time": iteration,
        "workmem_pairs": workmem_pairs if iteration % CONTENT_LOG_FREQUENCY == 0 else len(workmem_pairs), 
        "shortmem_pairs": shortmem_pairs,
        "longmem_pairs": longmem_pairs
    }
    logging.info(json.dumps(log_entry, default=serialize_to_json))

def extract_data_from_log(log_filename='memory_manager.log'):
    extracted_data = []

    with open(log_filename, 'r') as log_file:
        for line in log_file:
            try:
                log_entry = json.loads(line)
                if log_entry.get("action") == "save_state":
                    data = {
                        "time": log_entry["time"],
                        "workmem_pairs": log_entry["workmem_pairs"],
                        "shortmem_pairs": log_entry["shortmem_pairs"],
                        "longmem_pairs": log_entry["longmem_pairs"]
                    }
                    extracted_data.append(data)
            except json.JSONDecodeError:
                continue

    return extracted_data

def save_to_yaml(memory_manager, saving_path):
    yaml_file_path = os.path.join(saving_path, 'memory_bank_final.yaml')
    
    data = {
        'workmem': memory_manager.workmem.items,
        'shortmem': memory_manager.shortmem.items,
        'longmem': memory_manager.longmem.items
    }
    
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def plot_memory_sizes_over_time(filename, log_filename='memory_manager.log'):
    times = []
    shortmem_sizes = []
    longmem_sizes = []
    
    extracted_data = extract_data_from_log(log_filename)

    for entry in extracted_data:
        # Check if it's a list or integer
        recent_size = len(entry['shortmem_pairs']) if isinstance(entry['shortmem_pairs'], list) else entry['shortmem_pairs']
        long_size = len(entry['longmem_pairs']) if isinstance(entry['longmem_pairs'], list) else entry['longmem_pairs']

        times.append(entry['time'])
        shortmem_sizes.append(recent_size)
        longmem_sizes.append(long_size)
    
    plt.figure()
    plt.plot(times, shortmem_sizes, label='Short-Term Memory Size', color='blue')
    plt.plot(times, longmem_sizes, label='Long Memory Size', color='red')
    plt.xlabel('Time')
    plt.ylabel('Memory Size')
    plt.title('Memory Sizes as a Function of Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    
    plt.figure()
    plt.plot(np.arange(len(times)), times)
    
    return shortmem_sizes[-1], longmem_sizes[-1]

def plot_cosine_similarity_embeddings(filename, workmem_pairs, memory_manager):
    workmem_embeddings = np.array([embedding for _, embedding in workmem_pairs])
    workmem_embeddings = np.squeeze(workmem_embeddings, axis=1)
    longmem_embeddings = np.array(memory_manager.longmem.items_embeddings)

    dist_workmem_matrix = 1 - pairwise_distances(workmem_embeddings, metric='cosine')
    dist_workmem_upper = dist_workmem_matrix[np.triu_indices(dist_workmem_matrix.shape[0], k=1)]

    dist_longmem_matrix = 1 - pairwise_distances(longmem_embeddings, metric='cosine')
    dist_longmem_upper = dist_longmem_matrix[np.triu_indices(dist_longmem_matrix.shape[0], k=1)]

    plt.figure().tight_layout(pad=5.0)
    plt.hist(dist_workmem_upper, bins=50, alpha=0.5, label='workmem', density=True)
    plt.axvline(np.mean(dist_workmem_upper), color='blue', linewidth=2, linestyle='dotted')
    plt.hist(dist_longmem_upper, bins=50, alpha=0.5, label='longmem', density=True)
    plt.axvline(np.mean(dist_longmem_upper), color='orange', linewidth=2, linestyle='dotted')

    plt.xlabel('Cosine similarity')
    plt.ylabel('Probability density')
    plt.title('Distribution of cosine distances for different memory types')
    plt.legend(loc='upper right')
    plt.savefig(filename)