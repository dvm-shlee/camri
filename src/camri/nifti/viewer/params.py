class CrosslineParams(dict):
    """
    A dictionary-like structure that enforces specific keys.
    """
    allowed_keys = {"color", "width", "alpha", "style"}
    default_values = {
        "color": 'white',
        "width": 1,
        "alpha": 0.5,
        "style": '--'
    }
    
    def __setitem__(self, key, value):
        if key not in self.allowed_keys:
            raise KeyError(f"'{key}' is not a valid key. Allowed keys are: {self.allowed_keys}")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key not in self.allowed_keys:
            raise KeyError(f"'{key}' is not a valid key. Allowed keys are: {self.allowed_keys}")
        return super().__getitem__(key) or self.default_values[key]
