import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, zscore

class HypothesisTester:
    @staticmethod
    def chi_squared_test(data, col1, col2):
        """
        Perform a chi-squared test of independence between two categorical variables.
        """
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return {"chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof}

    @staticmethod
    def t_test(group1, group2):
        """
        Perform an independent t-test for two groups.
        """
        stat, p = ttest_ind(group1, group2, equal_var=False)
        return {"t_statistic": stat, "p_value": p}

    @staticmethod
    def z_test(group1, group2):
        """
        Perform a z-test for two groups.
        """
        diff = group1.mean() - group2.mean()
        pooled_std = ((group1.std()**2 + group2.std()**2) / 2) ** 0.5
        z_stat = diff / (pooled_std / (len(group1) ** 0.5))
        return {"z_statistic": z_stat}

# Example usage:
# tester = HypothesisTester()
# chi2_result = tester.chi_squared_test(data, 'gender', 'state')
# t_test_result = tester.t_test(group1, group2)
