import pytest
from clintrials.coll import to_1d_list

@pytest.mark.parametrize('value,expected', [
    (0, [0]),
    ([1, 2, 3], [1, 2, 3]),
    ([[1, 2], 3, [4, 5]], [1, 2, 3, 4, 5]),
    ([[1, [2, [3]]]], [1, 2, 3]),
    ((1, 2), [(1, 2)]),
])
def test_to_1d_list(value, expected):
    assert to_1d_list(value) == expected
