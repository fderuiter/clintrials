import collections

class SimulationResult:
    """Unified result container for simulation runs."""
    
    def __init__(self, results, mode="iterative"):
        self.results = results
        self.mode = mode
        
    def __iter__(self):
        return iter(self.results)
        
    def __len__(self):
        return len(self.results)
        
    def __getitem__(self, key_or_idx):
        return self.results[key_or_idx]
        
    def get(self, key, default=None):
        if isinstance(self.results, dict):
            return self.results.get(key, default)
        raise AttributeError("Underlying results are not a dictionary.")
        
    def keys(self):
        if isinstance(self.results, dict):
            return self.results.keys()
        raise AttributeError("Underlying results are not a dictionary.")
        
    def values(self):
        if isinstance(self.results, dict):
            return self.results.values()
        raise AttributeError("Underlying results are not a dictionary.")
        
    def items(self):
        if isinstance(self.results, dict):
            return self.results.items()
        raise AttributeError("Underlying results are not a dictionary.")
        
    def to_list(self):
        if isinstance(self.results, dict):
            return [self.results]
        return list(self.results)

