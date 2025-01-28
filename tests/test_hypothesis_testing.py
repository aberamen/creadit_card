import pandas as pd
from src.ab_testing.hypothesis_testing import HypothesisTester

def test_chi_squared_test():
    data = pd.DataFrame({
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'state': ['NY', 'CA', 'NY', 'CA', 'TX']
    })
    result = HypothesisTester.chi_squared_test(data, 'gender', 'state')
    assert 'chi2_statistic' in result
    assert 'p_value' in result

def test_t_test():
    group1 = [1, 2, 3, 4]
    group2 = [3, 4, 5, 6]
    result = HypothesisTester.t_test(group1, group2)
    assert 't_statistic' in result
    assert 'p_value' in result

if __name__ == "__main__":
    test_chi_squared_test()
    test_t_test()
    print("All tests passed!")
