def validate_matching_lengths(**kwargs):
    """Validates that all provided arrays have the same length.
    
    Pass arrays as keyword arguments. For example:
    validate_matching_lengths(array1=arr1, array2=arr2)
    """
    if not kwargs:
        return
    
    iterator = iter(kwargs.items())
    first_name, first_arr = next(iterator)
    expected_len = len(first_arr)
    
    for name, arr in iterator:
        if len(arr) != expected_len:
            raise ValueError(f"{first_name} and {name} should be same length.")

def validate_expected_length(array, expected_length: int, name: str):
    """Validates that an array has exactly the expected length."""
    if len(array) != expected_length:
        raise ValueError(f"{name} should have {expected_length} items.")

def validate_bounds(value, lower, upper, name: str, exclusive=False):
    """Validates that a numerical value is within the specified bounds."""
    if exclusive:
        if not (lower < value < upper):
            raise ValueError(f"{name} must be between {lower} and {upper}.")
    else:
        if not (lower <= value <= upper):
            raise ValueError(f"{name} must be between {lower} and {upper}.")

def validate_probability(value, name: str, exclusive=False):
    """Validates that a value is a valid probability between 0 and 1."""
    validate_bounds(value, 0, 1, name, exclusive=exclusive)

def validate_positive_integer(value, name: str):
    """Validates that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
