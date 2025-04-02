from enum import Enum

class DataType(Enum):
    """
    Enum for different types of data.
    """
    FUNDAMENTALS = "fundamentals"
    RETURNS = "returns"
    ALTERNATE = "alternate"

class DataStage(Enum):
    """
    Enum for different stages of data processing.
    """
    RAW = "raw"
    INTERMEDIATE = "intermediate"
    PROCESSED = "processed"
    FEATURES = "features" 