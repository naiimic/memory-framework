from omegaconf import OmegaConf
from settings import BASE_DIR
import random

def load_critical_mundane_events(num_critical_events = 5, num_mundane_events = 1000):
    EXPERIMENT_DIR = BASE_DIR / 'data'

    critical_events = OmegaConf.load(EXPERIMENT_DIR / 'critical_events.yaml')
    mundane_events = OmegaConf.load(EXPERIMENT_DIR / 'mundane_dailylife.yaml')

    critical_events = OmegaConf.to_container(critical_events, resolve=True)
    mundane_events = OmegaConf.to_container(mundane_events, resolve=True)

    critical_list = list(critical_events.values()) if isinstance(critical_events, dict) else critical_events
    critical_list = [random.sample(inner_list, num_critical_events) for inner_list in critical_list]
    mundane_list = list(mundane_events.values()) if isinstance(mundane_events, dict) else mundane_events
    mundane_list = [mundane_list[0] * 1000]
    mundane_list[0] = mundane_list[0][:num_mundane_events]

    merged_events = critical_list + mundane_list
    merged_events = [item for sublist in merged_events for item in sublist]
    random.shuffle(merged_events)

    return critical_list, mundane_list, merged_events