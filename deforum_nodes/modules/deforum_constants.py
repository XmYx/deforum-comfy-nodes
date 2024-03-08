class DeforumStorage:
    _instance = None  # Class attribute that holds the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DeforumStorage, cls).__new__(cls, *args, **kwargs)
            # Initialize your object here if needed
            cls._instance.deforum_models = {}
            cls._instance.deforum_cache = {}
            cls._instance.deforum_depth_algo = ""
            cls._instance.reset = None
        return cls._instance

    def __init__(self):
        # Initialization code here. If there are attributes that should not be reinitialized,
        # you can check if they already exist before setting them.
        pass