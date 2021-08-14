# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:41:38 2021

@author: Bigghost
"""

import unittest
from UnsupervisedValidation import *
from Utils import *


class InvalidMatrixTestCases():
    @staticmethod
    def get_invalid_type_message(argument_name, argument_order):
        return (argument_name.capitalize() + 
                " is not a numpy n-dimension array!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_type_pd_array_message(argument_name, argument_order):
        return ("The " + argument_name + " is not a Pandas Dataframe!\n" +  
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_not_nparray_message(argument_name, argument_order):
        return (argument_name.capitalize() + 
                " is not a numpy n-dimension array!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_shape_message(argument_name, argument_order):
        return ("The dimension of " + argument_name + " should be 2D!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_empty_row_message(argument_name, argument_order):
        return ("#Row of " + argument_name + " is zero!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_empty_column_message(argument_name, argument_order):
        return ("#Column of " + argument_name + " is zero!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_element_nan_message(argument_name, argument_order):
        return ("NaN values detected in " + argument_name + "!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_element_inf_message(argument_name, argument_order):
        return ("Infinite values detected in " + argument_name + "!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_element_not_numeric_message(argument_name, argument_order):
        return ("Non-numerical values detected in " + argument_name + "!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_element_too_low_message(argument_name, argument_order, lower_limit):
        return ("One or more values less than lower_limit (= " + 
                        str(lower_limit) + ") in " + argument_name + "!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_element_too_high_message(argument_name, argument_order, upper_limit):
        return ("One or more values higher than upper_limit (= " + 
                        str(upper_limit) + ") in " + argument_name + "!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_inconsistent_shape_two_matrix_message(matrix_name_A, matrix_name_B, argument_order):
        return ("The sizes of " + matrix_name_A + " and " + matrix_name_B + " are different!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_score_dimension_message(argument_name, argument_order):
        return("The " + argument_name + " should have only one column with name 'Score'!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_score_column_name_message(argument_name, argument_order):
        return("The " + argument_name + " should have only one column with name 'Score'!\n" + 
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def prepare_invalid_matrix_testcases(argument_name, argument_order, func,
                                         lower_limit = None, upper_limit = None): 
        test_case_list = [ None,
                          "Test",
                           np.array([[[1]]]),
                           np.array([[]]).transpose(),
                           np.array([[]]),
                           np.array([[np.nan]]),
                           np.array([[np.inf]]),
                           np.array([["TEST"]])]
        
        expected_error_type = [ TypeError,
                                TypeError,
                                ValueError,
                                ValueError,
                                ValueError,
                                ValueError,
                                ValueError,
                                TypeError]
        
        
        
        expected_output_list = [
        InvalidMatrixTestCases.get_not_nparray_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_not_nparray_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_shape_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_empty_row_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_empty_column_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_element_nan_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_element_inf_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_element_not_numeric_message(
            argument_name, argument_order)
        ]
        
        if lower_limit is not None:
            test_case_list.extend(np.array([[lower_limit - 1.0]]))
            expected_error_type.extend(ValueError)
            expected_output_list.extend(InvalidMatrixTestCases.get_element_too_low_message(
                argument_name, argument_order, lower_limit))
            
        if upper_limit is not None:
            test_case_list.extend(np.array([[upper_limit + 1.0]]))
            expected_error_type.extend(ValueError)
            expected_output_list.extend(InvalidMatrixTestCases.get_element_too_high_message(
                argument_name, argument_order, lower_limit))
        
        target_function = TargetFunction(
            argument_name,
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func
        )
        
        return target_function
    
    @staticmethod
    def prepare_invalid_pd_dataframe_testcases_typeonly(argument_name, argument_order, func): 
        
        test_case_list = [ None,
                          "Test"]
        
        expected_error_type = [ TypeError,
                                TypeError]
        
        expected_output_list = [
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        ]

        target_function = TargetFunction(
            argument_name,
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func)      
        
        return target_function
        
    @staticmethod
    def prepare_invalid_pd_dataframe_testcases(argument_name, argument_order, func,
                                               lower_limit = None, upper_limit = None): 
        invalid_pd_dataA = pd.DataFrame(1,index = ["Test",0.5, 1.0], columns = [1,2,3])
        invalid_pd_dataB = pd.DataFrame(1,index = [-0.1 ,0.5, 1.0], columns = [1,2,3])
        invalid_pd_dataC = pd.DataFrame(1,index = [0.0 ,0.5, 1.1], columns = [1,2,3])
        
        test_case_list = [ None,
                          "Test",
                           invalid_pd_dataA,
                           invalid_pd_dataB,
                           invalid_pd_dataC]
        
        expected_error_type = [ TypeError,
                                TypeError,
                                TypeError,
                                ValueError,
                                ValueError ]
        
        expected_output_list = [
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidNoiseRatioTestCases.get_invalid_type_message(argument_order),
        InvalidNoiseRatioTestCases.get_noise_ratio_too_low_message(argument_order),
        InvalidNoiseRatioTestCases.get_noise_ratio_too_high_message(argument_order),
        ]
        
        if lower_limit is not None:
            test_case_list.append(pd.DataFrame(lower_limit - 1.0,
                                               index = [0.0 ,0.5, 1.0], 
                                               columns = [1,2,3]))
            expected_error_type.append(ValueError)
            expected_output_list.append(InvalidMatrixTestCases.get_element_too_low_message(
                argument_name, argument_order, lower_limit))
            
        if upper_limit is not None:
            test_case_list.append(pd.DataFrame(upper_limit + 1.0,
                                               index = [0.0 ,0.5, 1.0], 
                                               columns = [1,2,3]))
            expected_error_type.append(ValueError)
            expected_output_list.append(InvalidMatrixTestCases.get_element_too_high_message(
                argument_name, argument_order, upper_limit))
        
        target_function = TargetFunction(
            argument_name,
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func)      
        
        return target_function
    
    
    @staticmethod
    def prepare_invalid_scores_testcases(argument_name, argument_order, func): 
        invalid_pd_dataA = pd.DataFrame(1,index = [0, 1, 2], columns = [1,2,3])
        invalid_pd_dataB = pd.DataFrame(1,index = [0, 1, 2], columns = ["Test"])
        
        test_case_list = [ None,
                          "Test",
                           invalid_pd_dataA,
                           invalid_pd_dataB]
        
        expected_error_type = [ TypeError,
                                TypeError,
                                ValueError,
                                ValueError ]
        
        expected_output_list = [
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_score_dimension_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_score_column_name_message(
            argument_name, argument_order)
        ]
        

        test_case_list.append(pd.DataFrame(-0.1,
                                           index = [0,1,2], 
                                           columns = ['Score']))
        expected_error_type.append(ValueError)
        expected_output_list.append(InvalidMatrixTestCases.get_element_too_low_message(
            argument_name, argument_order, 0.0))
            
        test_case_list.append(pd.DataFrame(1.1,
                                           index = [0,1,2], 
                                           columns = ['Score']))
        expected_error_type.append(ValueError)
        expected_output_list.append(InvalidMatrixTestCases.get_element_too_high_message(
            argument_name, argument_order, 1.0))
        
        target_function = TargetFunction(
            argument_name,
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func)      
        
        return target_function
        
    @staticmethod
    def prepare_inconsistent_matrix_shape_testcases(matrix_name_A, matrix_name_B,
                                                    argument_order, func):
        test_case_list = [np.random.rand(5,5), 
                          np.random.rand(20,5), 
                          np.random.rand(5,20)]
        
        expected_error_type = [ValueError,
                               ValueError,
                               ValueError]
        
        expected_output_list = [ 
            InvalidMatrixTestCases.get_inconsistent_shape_two_matrix_message(
                matrix_name_A, matrix_name_B,argument_order),
            InvalidMatrixTestCases.get_inconsistent_shape_two_matrix_message(
                matrix_name_A, matrix_name_B,argument_order),
            InvalidMatrixTestCases.get_inconsistent_shape_two_matrix_message(
                matrix_name_A, matrix_name_B,argument_order),
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
        
class InvalidNoiseRatioTestCases():
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("The noise ratio is not a float!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_noise_ratio_too_low_message(argument_order):
        return ("The noise ratio is too low: \n" + 
                "The noise ratio should not be lower than 0!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_noise_ratio_too_high_message(argument_order):
        return ("The noise ratio is too high: \n" + 
                "The noise ratio should not be higher than 1!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_noise_ratio_cases(argument_order, func):
        test_case_list = [None,
                           "Test",
                           -0.1,
                           1.1]
        
        expected_error_type = [TypeError,
                                TypeError,
                                ValueError,
                                ValueError ]
        
        expected_output_list = [
            InvalidNoiseRatioTestCases.get_invalid_type_message(argument_order),
            InvalidNoiseRatioTestCases.get_invalid_type_message(argument_order),
            InvalidNoiseRatioTestCases.get_noise_ratio_too_low_message(argument_order),
            InvalidNoiseRatioTestCases.get_noise_ratio_too_high_message(argument_order),
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
    

class InvalidNTrialTestCases():
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("The number of trial is not an integer!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_n_trial_negative_message(argument_order):
        return ("The number of trial should be greater than zero!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_trial_index_type_message(argument_order):
        return ("The trial index is not an integer!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_trial_index_negative_message(argument_order):
        return ("The trial index should not be negative!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_trial_index_too_large_message(argument_order):
        return ("The trial index should not be greater than or equal to #column of input result!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_n_trial_cases(argument_order, func):
        test_case_list = [None,
                           "Test",
                           0.1,
                           -1]
        
        expected_error_type = [TypeError,
                                TypeError,
                                TypeError,
                                ValueError ]
        
        expected_output_list = [
            InvalidNTrialTestCases.get_invalid_type_message(argument_order),
            InvalidNTrialTestCases.get_invalid_type_message(argument_order),
            InvalidNTrialTestCases.get_invalid_type_message(argument_order),
            InvalidNTrialTestCases.get_n_trial_negative_message(argument_order),
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
    
    @staticmethod
    def prepare_invalid_trial_index_cases(argument_order, func):
        test_case_list = [None,
                           "Test",
                           0.1,
                           -1,
                           1000]
        
        expected_error_type = [TypeError,
                                TypeError,
                                TypeError,
                                ValueError,
                                ValueError]
        
        expected_output_list = [
            InvalidNTrialTestCases.get_invalid_trial_index_type_message(argument_order),
            InvalidNTrialTestCases.get_invalid_trial_index_type_message(argument_order),
            InvalidNTrialTestCases.get_invalid_trial_index_type_message(argument_order),
            InvalidNTrialTestCases.get_trial_index_negative_message(argument_order),
            InvalidNTrialTestCases.get_trial_index_too_large_message(argument_order),
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
    

class InvalidNoiseRatioStepTestCases():
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("The noise ratio step is not a float!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_noise_ratio_step_too_low_message(argument_order):
        return ("The noise ratio step is too low: \n" + 
                "The noise ratio step should be greater than 0!\n" +  
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_noise_ratio_step_too_high_message(argument_order):
        return ("The noise ratio step is too high: \n" + 
                "The noise ratio step should not be higher than 0.5!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_noise_ratio_step_cases(argument_order, func):
        test_case_list = [None,
                           "Test",
                           -0.1,
                           0.0,
                           0.6]
        
        expected_error_type = [TypeError,
                                TypeError,
                                ValueError,
                                ValueError,
                                ValueError ]
        
        expected_output_list = [
            InvalidNoiseRatioStepTestCases.get_invalid_type_message(argument_order),
            InvalidNoiseRatioStepTestCases.get_invalid_type_message(argument_order),
            InvalidNoiseRatioStepTestCases.get_noise_ratio_step_too_low_message(argument_order),
            InvalidNoiseRatioStepTestCases.get_noise_ratio_step_too_low_message(argument_order),
            InvalidNoiseRatioStepTestCases.get_noise_ratio_step_too_high_message(argument_order),
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
    
class InvalidPValueThresholdTestCases():
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("The p-value threshold is not a float!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_pvalue_threshold_too_low_message(argument_order):
        return ("The p-value threshold is too low: \n" + 
                "The p-value threshold should not be lower than 0!\n" +
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_pvalue_threshold_too_high_message(argument_order):
        return ("The p-value threshold is too high: \n" + 
                "The p-value threshold should not be higher than 1!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_pvalue_threshold_cases(argument_order, func):
        test_case_list = [None,
                           "Test",
                           -0.1,
                           1.1]
        
        expected_error_type = [TypeError,
                                TypeError,
                                ValueError,
                                ValueError ]
        
        expected_output_list = [
            InvalidPValueThresholdTestCases.get_invalid_type_message(argument_order),
            InvalidPValueThresholdTestCases.get_invalid_type_message(argument_order),
            InvalidPValueThresholdTestCases.get_pvalue_threshold_too_low_message(argument_order),
            InvalidPValueThresholdTestCases.get_pvalue_threshold_too_high_message(argument_order),
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

    
class TargetFunction():
    """
    Test a function for one specific argument.
    Other argument will be hard-coded inside this class; Generally, those
    argument should be valid
    """
    
    def __init__(self, argument_name, argument_order, 
                 test_case_list, 
                 expected_error_type,
                 expected_output_list, 
                 func):
        
        self.argument_name = argument_name
        self.argument_order = argument_order
        self.test_case_list = test_case_list
        self.expected_error_type = expected_error_type
        self.expected_output_list = expected_output_list
        self.func = func
        
        if len(expected_error_type) > len(test_case_list):
            raise ValueError("#Expected Error Type > #Test Case!")
            
        if len(expected_output_list) > len(test_case_list):
            raise ValueError("#Expected Output > #Test Case!")
        
    def run(self, idx):
        self.func(self.test_case_list[idx])
        
    def get_test_case_num(self):
        return len(self.test_case_list)
        
    def get_expected_error_type(self, idx):
        return self.expected_error_type[idx]
    
    def get_expected_output_list(self, idx):
        return self.expected_output_list[idx]


class Test_Func_Base(unittest.TestCase):
    def base_test_matrix(self, target_function):
        for idx in range(target_function.get_test_case_num()):
            expected_error_type = target_function.get_expected_error_type(idx)
            expected_output = target_function.get_expected_output_list(idx)       
            
            if expected_error_type is not None:
                with self.assertRaises(target_function.get_expected_error_type(idx)) as context:
                    target_function.run(idx)
                
                the_exception = str(context.exception)
                self.assertEqual(the_exception, expected_output)
            else:
                result = target_function.run()
                self.assertEqual(result, expected_output)

        
class TestUtils(Test_Func_Base):
    def test_prepare_noise(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "input matrix", "first argument", 
                lambda x : Utils.prepare_noise(x))
        
        self.base_test_matrix(target_function)
                
                
if __name__ == '__main__':
    unittest.main()