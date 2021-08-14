# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:49:11 2021

@author: Bigghost
"""

import unittest
from CorrelationValidation import *
from Utils import *
from Unittest_Utils import *


class InvalidInputConditionMessage():
    @staticmethod
    def get_not_nparray_message(argument_order):
        return ("Input condition for correlation validation" +
                " is not a numpy n-dimension array!\n" +
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_shape_message(argument_order):
        return ("The dimension of input condition" + 
                " for correlation validation" + 
                " should be 1D!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_empty_message(argument_order):
        return ("The input condition for correlation validation is empty!\n" + 
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def get_only_one_condition_message(argument_order):
        return ("All samples are from the same conditions!\n" +
                "At least two conditions should be provided.\n" + 
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def get_all_samples_from_different_conditions_message(argument_order):
        return ("All samples are from different conditions!\n" +
                "At least two samples should be from the same conditions.\n" + 
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def get_invalid_element_num_message(argument_order):
        return ("The #element in the input condition for correlation validation" + 
                " is not equal to the #samples in the data matrix.\n" +
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def get_invalid_element_type_message(argument_order):
        return ("At least one element in the input condition " + 
                "for correlation validation is not a String!\n" + 
                Utils.generate_additional_description(argument_order))

class InvalidInputConditionTestCases():
    def prepare_invalid_input_condition_testcases(argument_order, func):
        
        test_case_list = [
            None,
            "Test",
            np.array([[1,2],[3,4]]),
            np.array([]),
            np.array(["1","1","1"]),
            np.array([("Cond" + str(x)) for x in range(20)]),
            np.array(["1","2","3","4","5"]),
            np.repeat(np.array([1,2,3,4]),5)
        ]
        
        expected_error_type = [
            TypeError,
            TypeError,
            ValueError,
            ValueError,
            ValueError,
            ValueError,
            ValueError,
            TypeError
        ]
        
        expected_output_list = [
            InvalidInputConditionMessage.get_not_nparray_message(
                argument_order),
            InvalidInputConditionMessage.get_not_nparray_message(
                argument_order),
            InvalidInputConditionMessage.get_invalid_shape_message(
                argument_order),
            InvalidInputConditionMessage.get_empty_message(
                argument_order),
            InvalidInputConditionMessage.get_only_one_condition_message(
                argument_order),
            InvalidInputConditionMessage.get_all_samples_from_different_conditions_message(
                argument_order),
            InvalidInputConditionMessage.get_invalid_element_num_message(
                argument_order),
            InvalidInputConditionMessage.get_invalid_element_type_message(
                argument_order),
        ]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func
        )
        
        return target_function
    
class GoodInputCases:
    @staticmethod
    def prepare_good_conditions(data_matrix):
        conditions = np.array([("Cond" + str(x)) for x in \
                          np.random.choice(
                              range(data_matrix.shape[1]),
                              data_matrix.shape[1])
                          ])
        return conditions
    
class ValidateCorrelationValidationResult:
    def validate_raw_result(pvalue, 
                            corr_all, corr_same_condition, corr_diff_condition):
        Utils.check_float(pvalue, "", "")
        Utils.check_value_boundary(pvalue < 0 and pvalue > 1, "", "") 
        
        test_list = [corr_all, corr_same_condition, corr_diff_condition]
        for x in test_list:
            for y in x :
                Utils.check_float(y, "", "")
                Utils.check_value_boundary(y < 0 and y > 1, "", "") 
        
class TestCorrelationValidationModule(Test_Func_Base):
    def test_correlation_evaluation_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "input matrix", 
                "first argument", 
                lambda x : CorrelationValidation.correlation_evaluation(
                            x, None))
        self.base_test_matrix(target_function)
        
    def test_correlation_evaluation_second_argument(self):
        good_matrix = np.random.rand(20,20)
        target_function = \
            InvalidInputConditionTestCases.prepare_invalid_input_condition_testcases(
                "second argument", 
                lambda x : CorrelationValidation.correlation_evaluation(
                            good_matrix, x))
        self.base_test_matrix(target_function)
        
    def test_correlation_evaluation_normal(self):
        for i in range(10):
            data_matrix = np.random.rand(20,50)
            conditions = GoodInputCases.prepare_good_conditions(data_matrix)
            
            pvalue, corr_all, \
            corr_same_condition, corr_diff_condition = \
                CorrelationValidation.correlation_evaluation(
                    data_matrix, conditions)
                
            ValidateCorrelationValidationResult.validate_raw_result(
                pvalue, corr_all, corr_same_condition, corr_diff_condition)
            
    def test_validate_data_matrix_specific_noise_ratio_first_argument(self):
        target_function = InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
            "input matrix", 
            "first argument", 
            lambda x : CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                x, None, None, None))
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_second_argument(self):
        good_matrix = np.random.rand(20,20)
        target_function = \
            InvalidInputConditionTestCases.prepare_invalid_input_condition_testcases(
                "second argument", 
                lambda x : CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                            good_matrix, x, None, None))
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_third_argument(self):
        good_matrix = np.random.rand(20,20)
        conditions = GoodInputCases.prepare_good_conditions(good_matrix)
            
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "noise matrix",
                "third argument", 
                lambda x : CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                            good_matrix, conditions, x, None))
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_fourth_argument(self):
        good_matrix = np.random.rand(20,20)
        good_matrix2 = np.random.rand(20,20)
        conditions = GoodInputCases.prepare_good_conditions(good_matrix)
        
        target_function = \
            InvalidNoiseRatioTestCases.prepare_invalid_noise_ratio_cases(
                "fourth argument", 
                lambda x : CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                            good_matrix, conditions, good_matrix2, x))
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_inconsistent_shape(self):
        good_matrix = np.random.rand(20,20)
        conditions = GoodInputCases.prepare_good_conditions(good_matrix)
        target_function = \
            InvalidMatrixTestCases.prepare_inconsistent_matrix_shape_testcases(
                "input matrix","noise matrix",
                "first and third argument", 
                lambda x : CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                            good_matrix, conditions, x, 0.25))
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_normal(self):
        shape_list = [20, 50, 100]
        noise_ratio_list = np.arange(0, 1.1, 0.1)
        
        for shape in shape_list:
            for noise_ratio in noise_ratio_list:
                good_matrix = np.random.rand(shape,20)
                good_matrix2 = np.random.rand(shape,20)
                conditions = GoodInputCases.prepare_good_conditions(good_matrix)
                
                pvalue, corr_all, corr_same_condition, corr_diff_condition = \
                    CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                                    good_matrix, conditions, good_matrix2, noise_ratio)
                    
                ValidateCorrelationValidationResult.validate_raw_result(
                    pvalue, corr_all, corr_same_condition, corr_diff_condition)
    
    def test_validate_data_matrix_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "input matrix", 
                "first argument", 
                lambda x : CorrelationValidation.validate_data_matrix(
                    x, 
                    None))
        self.base_test_matrix(target_function)
            
    def test_validate_data_matrix_second_argument(self):
        good_matrix = np.random.rand(20,20)
        target_function = \
            InvalidInputConditionTestCases.prepare_invalid_input_condition_testcases( 
                "second argument", 
                lambda x : CorrelationValidation.validate_data_matrix(
                    good_matrix, 
                    x))
        self.base_test_matrix(target_function)
            
    def test_validate_data_matrix_third_argument(self):
        good_matrix = np.random.rand(20,20)
        good_condition = GoodInputCases.prepare_good_conditions(good_matrix)
        target_function = \
            InvalidNoiseRatioStepTestCases.prepare_invalid_noise_ratio_step_cases(
                "third argument", 
                lambda x : CorrelationValidation.validate_data_matrix(
                    good_matrix, 
                    good_condition,
                    x))
        self.base_test_matrix(target_function)
    
    def test_validate_data_matrix_fourth_argument(self):
        good_matrix = np.random.rand(20,20)
        good_condition = GoodInputCases.prepare_good_conditions(good_matrix)
        target_function = \
            InvalidNTrialTestCases.prepare_invalid_n_trial_cases(
                "fourth argument", 
                lambda x : CorrelationValidation.validate_data_matrix(
                    good_matrix, 
                    good_condition,
                    0.25,
                    x))
        self.base_test_matrix(target_function)
        
    
    def test_validate_data_matrix_normal(self):
        shape_list = [20, 50, 100]
        noise_ratio_step_list = [0.25, 0.5]
        for shape in shape_list:
            for noise_ratio_step in noise_ratio_step_list:
                for n_trial in [1,3,5]:
                    good_matrix = np.random.rand(shape,20)
                    result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
                        n_corr, n_same_corr = \
                            CorrelationValidation.validate_data_matrix(
                            good_matrix, 
                            GoodInputCases.prepare_good_conditions(good_matrix),
                            noise_ratio_step,
                            n_trial              
                            )
                            
                    ResultUtils.check_correlation_validation_result(
                        result, "")
                    ResultUtils.check_general_result(
                        avg_corr, "", "", lower_limit = -1.0, upper_limit = 1.0)
                    ResultUtils.check_general_result(
                        avg_same_corr, "", "", lower_limit = -1.0, upper_limit = 1.0)
                    ResultUtils.check_general_result(
                        std_corr, "", "", lower_limit = 0.0, upper_limit = 2.0)
                    ResultUtils.check_general_result(
                        std_same_corr, "", "", lower_limit = 0.0, upper_limit = 2.0)
                   
                    Utils.check_int(n_corr,"","")
                    Utils.check_value_boundary(n_corr <= 0, "","")
                    
                    Utils.check_int(n_same_corr, "", "")
                    Utils.check_value_boundary(n_same_corr <= 0, "", "")
                    
                    Utils.check_value_boundary(n_corr < n_same_corr, "", "")
        
            
    def test_plot_result(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "correlation validation result", 
                "first argument", 
                lambda x : CorrelationValidation.plot_result(x, "Test.svg", format="svg"), 
                lower_limit = 0.0, upper_limit = 1.0)
        self.base_test_matrix(target_function)
            
    def test_plot_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        CorrelationValidation.plot_result(test_pd_dataframeA, "Test.svg", format="svg")
        
    def test_parse_result_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "correlation validation result", 
                "first argument", 
                lambda x : CorrelationValidation.parse_result(x, None),
                lower_limit = 0.0, upper_limit = 1.0)
        self.base_test_matrix(target_function)
                
    def test_parse_result_second_argument(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        target_function = \
            InvalidPValueThresholdTestCases.prepare_invalid_pvalue_threshold_cases(
                "second argument", 
                lambda x : CorrelationValidation.parse_result(
                    test_pd_dataframeA, x))
        self.base_test_matrix(target_function)
            
    def test_parse_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        scores = CorrelationValidation.parse_result(test_pd_dataframeA, 0.05)
        ResultUtils.check_general_score(scores, "", "")
        
    
    def test_plot_avg_corr_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "correlation validation result", 
                "first argument", 
                lambda x : CorrelationValidation.plot_avg_corr(x, 
                  None, None, None, None, None,\
                  None, None, '0', 'correlation_avg_corr.svg', format='svg'),
                lower_limit = 0.0, upper_limit = 1.0)
        self.base_test_matrix(target_function)
                
    def test_plot_avg_corr_second_argument(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_scores_testcases(
                "correlation validation score", 
                "second argument", 
                lambda x : CorrelationValidation.plot_avg_corr(
                    test_pd_dataframeA, 
                    x, 
                    None, None, None, None,\
                    None, None, '0', 'correlation_avg_corr.svg', format='svg'))
        self.base_test_matrix(target_function)
                
    def test_plot_avg_corr_third_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "average correlation matrix (all condition)", 
                "third argument", 
                lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    x, None, None, None,\
                    None, None, '0', 'correlation_avg_corr.svg', format='svg'))
        self.base_test_matrix(target_function)
                
    def test_plot_avg_corr_fourth_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "average correlation matrix (same condition)", 
                "fourth argument", 
                lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, x, None, None,\
                    None, None, '0', 'correlation_avg_corr.svg', format='svg'))
        self.base_test_matrix(target_function)
    
    def test_plot_avg_corr_fifth_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "std. of correlation matrix (all conditions)", 
                "fifth argument", 
                lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, avg_same_corr, x, None,\
                    None, None, '0', 'correlation_avg_corr.svg', format='svg'))
        self.base_test_matrix(target_function)
        
    def test_plot_avg_corr_6th_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "std. of correlation matrix (same conditions)", 
                "6th argument", 
                lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, avg_same_corr, std_corr, x,\
                    None, None, '0', 'correlation_avg_corr.svg', format='svg'))
        self.base_test_matrix(target_function)
        
    def test_plot_avg_corr_7th_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        argument_order = "7th argument"
        
        test_case_list = [None,
                           "Test",
                           -0.1,
                           0,
                           -1]
        
        expected_error_type = [TypeError,
                                TypeError,
                                TypeError,
                                ValueError,
                                ValueError]
        
        expected_output_list = [
            "#correpation pairs of the entire dataset should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs of the entire dataset should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs of the entire dataset should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs of the entire dataset should be positive!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs of the entire dataset should be positive!" + 
            Utils.generate_additional_description(argument_order),
        ]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, avg_same_corr, std_corr, std_same_corr,\
                    x, None, '0', 'correlation_avg_corr.svg', format='svg')
            )
        self.base_test_matrix(target_function)
        
    def test_plot_avg_corr_8th_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        argument_order = "8th argument"
        
        test_case_list = [None,
                           "Test",
                           -0.1,
                           0,
                           -1]
        
        expected_error_type = [TypeError,
                                TypeError,
                                TypeError,
                                ValueError,
                                ValueError]
        
        expected_output_list = [
            "#correpation pairs from the same conditions should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs from the same conditions should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs from the same conditions should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs from the same conditions should be positive!" + 
            Utils.generate_additional_description(argument_order),
            
            "#correpation pairs from the same conditions should be positive!" + 
            Utils.generate_additional_description(argument_order),
        ]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, avg_same_corr, std_corr, std_same_corr,\
                    n_corr, x, '0', 'correlation_avg_corr.svg', format='svg')
            )
        self.base_test_matrix(target_function)
        
    def test_plot_avg_corr_7th_8th_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        
        argument_order = "7th and 8th argument"
        
        test_case_list = [n_corr]
        
        expected_error_type =  [ValueError]
        
        expected_output_list = [
            "#correpation pairs of the entire dataset should be greater than " + 
            "#correlation pairs from the same conditions"+ 
            Utils.generate_additional_description(argument_order),
            
        ]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, avg_same_corr, std_corr, std_same_corr,\
                    n_same_corr - 1, n_same_corr, '0', 'correlation_avg_corr.svg', format='svg')
            )
        self.base_test_matrix(target_function)
    
    def test_plot_avg_corr_9th_argument(self):
        good_matrix = np.random.rand(20,20)
        result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
            n_corr, n_same_corr = \
                CorrelationValidation.validate_data_matrix(
                good_matrix, 
                GoodInputCases.prepare_good_conditions(good_matrix),
                0.5,
                1              
                )
        scores = CorrelationValidation.parse_result(result, 0.05)
        target_function = InvalidNTrialTestCases.prepare_invalid_trial_index_cases(
            "9th argument", 
            lambda x : CorrelationValidation.plot_avg_corr(
                    result, 
                    scores, 
                    avg_corr, avg_same_corr, std_corr, std_same_corr,\
                    n_corr, n_same_corr, x, 'correlation_avg_corr.svg', format='svg'))
            
        self.base_test_matrix(target_function)
    
    def test_plot_avg_corr_normal(self):
        shape_list = [20, 50, 100]
        noise_ratio_step_list = [0.25, 0.5]
        for shape in shape_list:
            for noise_ratio_step in noise_ratio_step_list:
                for n_trial in [1,3,5]:
                    good_matrix = np.random.rand(shape,20)
                    result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
                        n_corr, n_same_corr = \
                            CorrelationValidation.validate_data_matrix(
                            good_matrix, 
                            GoodInputCases.prepare_good_conditions(good_matrix),
                            noise_ratio_step,
                            n_trial              
                            )
                            
                    scores = CorrelationValidation.parse_result(result, 0.05)
                    CorrelationValidation.plot_avg_corr(
                                result, 
                                scores, 
                                avg_corr, avg_same_corr, std_corr, std_same_corr,\
                                n_corr, n_same_corr, n_trial - 1, 'correlation_avg_corr.svg', format='svg')

if __name__ == '__main__':
    unittest.main()
