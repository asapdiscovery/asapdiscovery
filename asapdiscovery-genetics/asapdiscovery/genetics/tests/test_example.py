from asapdiscovery.genetics.example import example



def test_example(example_fixture):
    assert example() == example_fixture