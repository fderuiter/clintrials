"""Validation functions for the Clinical Trials library."""

from clintrials.core.errors import ErrorTemplates


def validate_matching_lengths(**kwargs):
    """Validates that all provided arrays have the same length.

    Pass arrays as keyword arguments. For example:
    validate_matching_lengths(array1=arr1, array2=arr2)

    Args:
        **kwargs: Arbitrary keyword arguments where keys are the names
            of the arrays and values are the arrays themselves.

    Returns:
        None

    Raises:
        ValueError: If any array does not match the length of the first array.
    """
    if not kwargs:
        return

    iterator = iter(kwargs.items())
    first_name, first_arr = next(iterator)
    expected_len = len(first_arr)

    for name, arr in iterator:
        if len(arr) != expected_len:
            raise ValueError(ErrorTemplates.MATCHING_LENGTHS.format(first_name=first_name, name=name))


def validate_expected_length(array, expected_length: int, name: str):
    """Validates that an array has exactly the expected length.

    Args:
        array (list or numpy.ndarray): The array to validate.
        expected_length (int): The expected length of the array.
        name (str): The name of the parameter being validated, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the array length does not match the expected length.
    """
    if len(array) != expected_length:
        raise ValueError(ErrorTemplates.EXPECTED_LENGTH.format(name=name, expected_length=expected_length))


def validate_bounds(value, lower, upper, name: str, exclusive=False):
    """Validates that a numerical value is within the specified bounds.

    Args:
        value (float or int): The numerical value to validate.
        lower (float or int): The lower bound.
        upper (float or int): The upper bound.
        name (str): The name of the parameter, used in the error message.
        exclusive (bool, optional): If True, bounds are exclusive (value > lower and value < upper).
            If False, bounds are inclusive (value >= lower and value <= upper). Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If the value is outside the specified bounds.
    """
    if exclusive:
        if value <= lower:
            raise ValueError(ErrorTemplates.GT.format(name=name, bound=lower))
        if value >= upper:
            raise ValueError(ErrorTemplates.LT.format(name=name, bound=upper))
    else:
        if value < lower:
            raise ValueError(ErrorTemplates.GE.format(name=name, bound=lower))
        if value > upper:
            raise ValueError(ErrorTemplates.LE.format(name=name, bound=upper))


def validate_probability(value, name: str, exclusive=False):
    """Validates that a value is a valid probability between 0 and 1.

    Args:
        value (float): The probability value to validate.
        name (str): The name of the parameter, used in the error message.
        exclusive (bool, optional): If True, probabilities of exactly 0 or 1 are invalid.
            Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If the value is not a valid probability.
    """
    if exclusive:
        if not (0 < value < 1):
            raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))
    else:
        if not (0 <= value <= 1):
            raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))


def validate_positive_integer(value, name: str):
    """Validates that a value is a positive integer.

    Args:
        value (int): The value to validate.
        name (str): The name of the parameter, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the value is not an integer or is less than or equal to zero.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(ErrorTemplates.POSITIVE_INTEGER.format(name=name))

import re


def parse_pdf_structure(pdf_bytes: bytes) -> dict:
    """Parses a generated PDF to extract its logical structure tree.

    Returns a dictionary of structure elements.

    Args:
        pdf_bytes (bytes): The byte string of the PDF content.

    Returns:
        dict: A dictionary of extracted logical structure elements.
    """
    content = pdf_bytes.decode('latin1', errors='ignore')
    objects = {}

    for match in re.finditer(r'(\d+)\s+0\s+obj(.*?)endobj', content, re.DOTALL):
        obj_id = int(match.group(1))
        objects[obj_id] = match.group(2)

    struct_elems = {}
    for obj_id, obj_content in objects.items():
        if '/Type /StructElem' in obj_content:
            s_match = re.search(r'/S\s+/(\w+)', obj_content)
            s_type = s_match.group(1) if s_match else None

            p_match = re.search(r'/P\s+(\d+)\s+0\s+R', obj_content)
            parent_id = int(p_match.group(1)) if p_match else None

            k_match = re.search(r'/K\s+\[(.*?)\]', obj_content, re.DOTALL)
            kids = []
            if k_match:
                k_content = k_match.group(1)
                kids_refs = re.findall(r'(\d+)\s+0\s+R', k_content)
                if kids_refs:
                    kids = [int(x) for x in kids_refs]
                else:
                    kids_ints = re.findall(r'\b\d+\b', k_content)
                    if kids_ints:
                        kids = [f"MCID_{x}" for x in kids_ints]

            struct_elems[obj_id] = {
                'id': obj_id,
                'type': s_type,
                'parent': parent_id,
                'kids': kids
            }

    return struct_elems

def validate_pdf_ua_structure(pdf_bytes: bytes):
    """Validates that a PDF's structure tree is correctly nested and MCIDs are only on leaves.

    Args:
        pdf_bytes (bytes): The byte string of the PDF content.

    Returns:
        bool: True if the structure is correctly nested.

    Raises:
        ValueError: If the structural nesting is incorrect or MCIDs are on containers.
    """
    elems = parse_pdf_structure(pdf_bytes)

    # Validation 1: Nesting correctness
    for obj_id, elem in elems.items():
        if elem['type'] == 'TR':
            parent = elems.get(elem['parent'])
            if parent and parent['type'] not in ('Table', 'THead', 'TBody', 'TFoot'):
                raise ValueError(f"TR (id {obj_id}) must be a child of a Table element")

        if elem['type'] in ('TH', 'TD'):
            parent = elems.get(elem['parent'])
            if parent and parent['type'] != 'TR':
                raise ValueError(f"{elem['type']} (id {obj_id}) must be a child of a TR element")

        # Validation 2: MCIDs are not on structural containers
        if elem['type'] in ('Table', 'TR', 'THead', 'TBody', 'TFoot'):
            for kid in elem['kids']:
                if isinstance(kid, str) and kid.startswith('MCID_'):
                    raise ValueError(f"Structural container {elem['type']} (id {obj_id}) cannot have MCID assigned directly")

    return True
