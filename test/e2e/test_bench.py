import re
from test.conftest import skip_if_no_llama_bench
from test.e2e.utils import check_call

import pytest


@pytest.mark.e2e
@skip_if_no_llama_bench
def test_model_and_params_columns(test_model):
    result = check_call(["ramalama", "bench", "-t", "4", "--ngl", "0", test_model])

    assert False
