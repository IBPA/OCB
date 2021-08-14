# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:49:11 2021

@author: Bigghost
"""

import unittest
from UnsupervisedValidation import *
from Utils import *
from Unittest_Utils import *

from missingpy import MissForest, KNNImputer

class InvalidDropRatioTestCases():
    #staticmethod
    def get_drop_ratio_not_float_message(argument_order):
        return ("The drop ratio is not a number!\n" + 
                Utils.generate_additional_description(argument_order))
            
    #staticmethod
    def get_drop_ratio_too_low_message(argument_order):
        return ("The drop ratio is too low or the input matrix may be too small:\n" + 
                "The number of drop values is less than or equal to zero!\n" + 
                Utils.generate_additional_description(argument_order))
            
    #staticmethod
    def get_drop_ratio_too_high_message(argument_order):
        return ("The drop ratio is too high:\n" + 
                "The number of drop values is high than 0.5!\n" + 
                Utils.generate_additional_description(argument_order))
            
    #staticmethod
    def get_drop_ratio_all_value_dropped_message(argument_order):
        return ("All value will be dropped:\n" + 
                "The size of the input matrix may be too small!\n" + 
                Utils.generate_additional_description(argument_order))
                
    #staticmethod
    def prepare_invalid_drop_ratio_testcases(argument_order, func):
        
        test_case_list = ["Test",
                          0,
                          0.01,
                          0.8]
        
        expected_error_type = [TypeError,
                                ValueError,
                                ValueError,
                                ValueError,
                              ]
        
        expected_output_list = [
            InvalidDropRatioTestCases.get_drop_ratio_not_float_message(argument_order),
            InvalidDropRatioTestCases.get_drop_ratio_too_low_message(argument_order),
            InvalidDropRatioTestCases.get_drop_ratio_too_low_message(argument_order),
            InvalidDropRatioTestCases.get_drop_ratio_too_high_message(argument_order)
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
            
class InvalidDroppedIndicesTestCases():
    #staticmethod
    def get_not_a_list_message(argument_order):
        return ("The indices list of dropped values is not a list!\n" +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_invalid_length_dropped_indices(argument_order):
        return ("The length of the indices list of dropped values is not equal to the #column of matrix before dropping values\n" +
                Utils.generate_additional_description(argument_order))
        
    #staticmethod
    def get_invalid_element_type_message(argument_order):
        return ("At least one element of the indices list of dropped values is not a numpy array!\n" +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_not_1D_nparray_message(argument_order):
        return ("At least one element of the indices list of dropped values is not a 1D numpy array!\n" +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_empty_message(argument_order):
        return ("At least one element of the indices list of dropped values is empty!\n" +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_too_long_message(argument_order):
        return ("At least one element of the indices list of dropped values is longer than the #row of imputed matrix!\n" +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_duplicated_message(argument_order):
        return ("At least one element of the indices list of dropped values contains duplicated values!\n" +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_has_nan_message(argument_order):
        return ("At least one element of the indices list of dropped "+ 
                        "values contains NaN values!\n"  +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_has_inf_message(argument_order):
        return ("At least one element of the indices list of dropped values contains infinite values!\n"  +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_has_non_numerical_message(argument_order):
        return ("At least one element of the indices list of dropped values contains non numerical values!\n"  +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_has_negative_value_message(argument_order):
        return ("At least one element of the indices list of dropped values contains negative values!\n"  +
                Utils.generate_additional_description(argument_order))
    
    #staticmethod
    def get_element_has_out_of_range_value_message(argument_order):
        return ("At least one element of the indices list of dropped values contains a value greater than #row of imputed matrix!\n"  +
                Utils.generate_additional_description(argument_order))

    #staticmethod
    def prepare_invalid_dropped_indices_testcases(argument_order, func):
        #assume the matrix size is (20, 20)
        test_case_list = [ None,                                #wrong type
                           "Test",                              #wrong type
                           [np.array([0])] * 100,               #wrong size
                           ["Test"] * 20,                       #wrong element type
                           [np.random.rand(5,5)] * 20,          #not 1D nparray
                           [np.array([])] * 20,                 #empty
                           [np.random.rand(100)] * 20,          #too long
                           [np.array([1,1,2])] * 20,            #duplicated
                           [np.array([np.nan])] * 20,           #nan
                           [np.array([np.inf])] * 20,           #inf
                           [np.array(["Test"])] * 20,           #not numerical
                           [np.array([-1,1,2])] * 20,           #negative
                           [np.array([20])] * 20]               #out of bound value
        
        expected_error_type = [TypeError,                       #wrong type
                                TypeError,                      #wrong type
                                ValueError,                     #wrong size
                                TypeError,                      #wrong element type
                                TypeError,                      #not 1D nparray
                                ValueError,                     #empty
                                ValueError,                     #too long
                                ValueError,                     #duplicated
                                ValueError,                     #nan
                                ValueError,                     #inf
                                ValueError,                     #not numerical
                                ValueError,                     #negative
                                ValueError]                     #out of bound value
        
        expected_output_list = [
            InvalidDroppedIndicesTestCases.get_not_a_list_message(argument_order),
            InvalidDroppedIndicesTestCases.get_not_a_list_message(argument_order),
            InvalidDroppedIndicesTestCases.get_invalid_length_dropped_indices(argument_order),
            InvalidDroppedIndicesTestCases.get_invalid_element_type_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_not_1D_nparray_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_empty_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_too_long_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_duplicated_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_has_nan_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_has_inf_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_has_non_numerical_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_has_negative_value_message(argument_order),
            InvalidDroppedIndicesTestCases.get_element_has_out_of_range_value_message(argument_order)
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
            
class ValidateDroppedMatrix():
    #staticmethod
    def validate(dropped_matrix, dropped_indices, dropped_ratio):
        if dropped_matrix.shape[1] != len(dropped_indices):
            raise ValueError("#col of dropped matrix != len(dropped indices)!")
            
        for col in range(dropped_matrix.shape[1]):
            dropped_number = 0
            for row in range(dropped_matrix.shape[0]):
                if row in dropped_indices[col]:
                    dropped_number += 1
                    if not np.isnan(dropped_matrix[row,col]):
                        raise ValueError("At least one element expected to be dropped still have a value!")
                else:
                    if np.isnan(dropped_matrix[row,col]):
                        raise ValueError("At least one value unexpectly dropped!")
            
            if dropped_number != np.round(dropped_matrix.shape[0] * dropped_ratio):
                raise ValueError("#Dropped values are not as expected!\n" + 
                                 "Expected = " + np.round(dropped_matrix.shape[0] * dropped_ratio) +
                                 "Dropped = " + dropped_number)
                
class InvalidImputerMessage():
    #staticmethod
    def get_invalid_inputer_message(argument_order):
        return ("The type of the imputer is invalid.\n" + 
                "The imputer should be missingpy.knnimpute.KNNImputer or" +
                "missingpy.missforest.MissForest.\n" + 
                Utils.generate_additional_description(argument_order))
                
    def prepare_invalid_imputer_testcases(argument_order, func):
        test_case_list = [None,
                          "Test"]
        
        expected_error_type = [TypeError, 
                               TypeError]
        
        expected_output_list = [ 
            InvalidImputerMessage.get_invalid_inputer_message(argument_order),
            InvalidImputerMessage.get_invalid_inputer_message(argument_order),
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

class TestUnsupervisedValidationModule(Test_Func_Base):
    def test_drop_valuesfunc_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "input data matrix", 
                "first argument", 
                lambda x : UnsupervisedValidation.drop_values(x, 0.5))
        self.base_test_matrix(target_function)
        
    def test_drop_valuesfunc_second_argument(self):
        target_function = InvalidDropRatioTestCases.prepare_invalid_drop_ratio_testcases(
            "second argument",
            lambda x: UnsupervisedValidation.drop_values(np.random.rand(5,5), x))

        self.base_test_matrix(target_function)
        
    def test_drop_valuesfunc_normal(self):
        size_list = [5, 10, 20, 100]
        test_ratio_list = [0.11, 0.3, 0.5] #np.round(0.5) is 0!
        for size in size_list:
            for test_ratio in test_ratio_list:
                dropped_matrix, dropped_indices = \
                UnsupervisedValidation.drop_values(
                    np.random.rand(size,size), test_ratio)
                ValidateDroppedMatrix.validate(
                    dropped_matrix, dropped_indices, test_ratio)
                
    def test_evaluate_impute_error_first_argument(self):
        good_matrix = np.random.rand(20,20) 
        dropped_matrix, dropped_indices = UnsupervisedValidation.drop_values(good_matrix, 0.5)
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "matrix before dropping values", 
                "first argument", 
                lambda x : UnsupervisedValidation.evaluate_impute_error(
                    x, good_matrix, dropped_indices))
    
        self.base_test_matrix(target_function)           
        
    def test_evaluate_impute_error_second_argument(self):
        good_matrix = np.random.rand(20,20) 
        dropped_matrix, dropped_indices = UnsupervisedValidation.drop_values(good_matrix, 0.5)
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "imputed matrix", 
                "second argument", 
                lambda x : UnsupervisedValidation.evaluate_impute_error(
                        good_matrix, x, dropped_indices))
    
        self.base_test_matrix(target_function)         
        
    def test_evaluate_impute_error_inconsistent_matrix_shape(self):
        good_matrix = np.random.rand(20,20) 
        dropped_matrix, dropped_indices = UnsupervisedValidation.drop_values(good_matrix, 0.5)
        
        target_function = InvalidMatrixTestCases.prepare_inconsistent_matrix_shape_testcases(
            "matrix before dropping values","the imputed matrix",
            "first two arguments",
            lambda x : UnsupervisedValidation.evaluate_impute_error(
                x, good_matrix, dropped_indices))
        
        self.base_test_matrix(target_function)  
        
    def test_evaluate_impute_invalid_dropped_indices(self):
        good_matrix = np.random.rand(20,20)
        
        target_function = \
            InvalidDroppedIndicesTestCases.prepare_invalid_dropped_indices_testcases(
                "third argument",
                lambda x : UnsupervisedValidation.evaluate_impute_error(
                    good_matrix, good_matrix, x))
        
        self.base_test_matrix(target_function)  
        
    def test_evaluate_impute_normal(self):
        size_list = [5, 10, 20, 100]
        test_ratio_list = [0.11, 0.3, 0.5] #np.round(0.5) is 0!
        for size in size_list:
            for test_ratio in test_ratio_list:
                good_matrix = np.random.rand(size, size)
                dropped_matrix, dropped_indices = \
                UnsupervisedValidation.drop_values(
                    good_matrix, test_ratio)
                
                error = UnsupervisedValidation.evaluate_impute_error(
                    good_matrix, good_matrix + 1, dropped_indices)
                
                self.assertAlmostEqual(error, 1)
                
                error = UnsupervisedValidation.evaluate_impute_error(
                    good_matrix, good_matrix - 1, dropped_indices)
                
                self.assertAlmostEqual(error, 1)
                
    def test_validate_data_matrix_specific_noise_ratio_first_argument(self):
        good_matrix = np.random.rand(20,20)
        good_matrix2 = np.random.rand(20,20)
        
        target_function = InvalidImputerMessage.prepare_invalid_imputer_testcases(
            "first argument",
            lambda x : UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                    x,
                    good_matrix,
                    good_matrix2,
                    0.5,
                    0.5))
                    
        self.base_test_matrix(target_function) 
        
    
    def test_validate_data_matrix_specific_noise_ratio_second_argument(self):
        good_matrix = np.random.rand(20,20) 
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "input data matrix", 
                "second argument", 
                lambda x : UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                    KNNImputer(),
                    x,
                    good_matrix,
                    0.5,
                    0.5))
    
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_third_argument(self):
        good_matrix = np.random.rand(20,20) 
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "noise matrix", 
                "third argument", 
                lambda x : UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                            KNNImputer(),
                            good_matrix,
                            x,
                            0.5,
                            0.5))

        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_fourth_argument(self):
        good_matrix = np.random.rand(20,20)
        good_matrix2 = np.random.rand(20,20)
        
        target_function = \
            InvalidNoiseRatioTestCases.prepare_invalid_noise_ratio_cases(
                "fourth argument",
                lambda x : UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                            KNNImputer(),
                            good_matrix,
                            good_matrix2,
                            x,
                            0.5))
        
        self.base_test_matrix(target_function)
        
    
    def test_validate_data_matrix_specific_noise_ratio_fifth_argument(self):
        good_matrix = np.random.rand(20,20)
        good_matrix2 = np.random.rand(20,20)
        
        target_function = InvalidDropRatioTestCases.prepare_invalid_drop_ratio_testcases(
            "fifth argument",
            lambda x : UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                            KNNImputer(),
                            good_matrix,
                            good_matrix2,
                            0.5, 
                            x))
        
        self.base_test_matrix(target_function) 
                    
    def test_validate_data_matrix_first_argument(self):
        good_matrix = np.random.rand(20,20)
        
        target_function = InvalidImputerMessage.prepare_invalid_imputer_testcases(
            "first argument",
            lambda x : UnsupervisedValidation.validate_data_matrix(
                            x,
                            good_matrix,
                            100,
                            0.1,
                            0.5))
    
        self.base_test_matrix(target_function) 
    
    def test_validate_data_matrix_second_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
                "input data matrix", 
                "second argument", 
                lambda x : UnsupervisedValidation.validate_data_matrix(
                                KNNImputer(),
                                x,
                                100,
                                0.1,
                                0.5))

        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_third_argument(self):
        good_matrix = np.random.rand(20,20)
        
        target_function = \
            InvalidNTrialTestCases.prepare_invalid_n_trial_cases(
                "third argument",
                lambda x : UnsupervisedValidation.validate_data_matrix(
                            KNNImputer(),
                            good_matrix,
                            x,
                            0.1,
                            0.5))
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_fourth_argument(self):
        good_matrix = np.random.rand(20,20)
        
        target_function = \
            InvalidNoiseRatioStepTestCases.prepare_invalid_noise_ratio_step_cases(
                "fourth argument",
                lambda x : UnsupervisedValidation.validate_data_matrix(
                            KNNImputer(),
                            good_matrix,
                            100,
                            x,
                            0.5))
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_fifth_argument(self):
        good_matrix = np.random.rand(20,20)
        
        target_function = InvalidDropRatioTestCases.prepare_invalid_drop_ratio_testcases(
                "fifth argument",
                lambda x : UnsupervisedValidation.validate_data_matrix(
                            KNNImputer(),
                            good_matrix,
                            100,
                            0.1,
                            x))
                            
        self.base_test_matrix(target_function)
        
    def test_read_result(self):
        target_function = \
        InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
            "unsupervised validation result", 
            "first argument", 
            lambda x : UnsupervisedValidation.read_result(x))

        self.base_test_matrix(target_function)
                
    def test_plot_result(self):
        target_function = \
        InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
            "unsupervised validation result", 
            "first argument", 
            lambda x : UnsupervisedValidation.plot_result(x, "test.png", format = ".svg"))

        self.base_test_matrix(target_function)
        
    def test_plot_single_result_first_argument(self):
        target_function = \
        InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
            "unsupervised validation result", 
            "first argument", 
            lambda x : UnsupervisedValidation.plot_single_result(x, 0, "test.png", format = ".svg"))

        self.base_test_matrix(target_function)
        
    def test_plot_single_result_second_argument(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        
        target_function = \
            InvalidNTrialTestCases.prepare_invalid_trial_index_cases(
                "second argument",
                lambda x : UnsupervisedValidation.plot_single_result(
                    test_pd_dataframeA, x, "test.png", format = ".svg") )
        
        self.base_test_matrix(target_function)
    
    
    def test_validate_data_matrix_specific_noise_ratio_normal(self):
        imputer_list = [KNNImputer(), MissForest(n_estimators = 1, max_depth=2)]
        size_list = [20]
        test_noise_ratio_list = [0.0, 0.5, 1.0]
        test_drop_ratio_list = [0.11, 0.3, 0.5] #np.round(0.5) is 0!
        for imputer in imputer_list:
            for size in size_list:
                for test_noise_ratio in test_noise_ratio_list:
                    for test_drop_ratio in test_drop_ratio_list:
                        error, mixed_matrix, imputed_matrix, dropped_indices = \
                            UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                            imputer, 
                            np.random.rand(size, size), np.random.rand(size, size),
                            test_noise_ratio,
                            test_drop_ratio)
                        
                        if not np.issubdtype(error, np.floating):
                            raise TypeError("The average error is not a number!")
                            
                        MatrixUtils.check_input_matrix(mixed_matrix, 
                                                 "mixed_matrix", "second output")
                        MatrixUtils.check_input_matrix(imputed_matrix, 
                                                 "imputed_matrix", "third output")
                        
                        if mixed_matrix.shape != imputed_matrix.shape:
                            raise ValueError("Mixed matrix size != Imputed matrix size!")
                        
                        DroppedIndicesUtils.check(
                            dropped_indices, mixed_matrix, "fourth output")
              
    
    def test_validate_data_matrix_normal(self):
        imputer_list = [KNNImputer(), MissForest(n_estimators = 1, max_depth=2)]
        size_list = [20]
        test_noise_ratio_step_list = [0.25, 0.5]
        test_drop_ratio_list = [0.5] #np.round(0.5) is 0!
        for imputer in imputer_list:
            for size in size_list:
                for test_noise_ratio_step in test_noise_ratio_step_list:
                    for test_drop_ratio in test_drop_ratio_list:
                        result = UnsupervisedValidation.validate_data_matrix(
                            imputer, 
                            np.random.rand(size, size), 
                            3,
                            test_noise_ratio_step,
                            test_drop_ratio)
                        
                        ResultUtils.check_unsupervised_validation_result(result, "")
                        
      
             
    def test_read_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        area_table = UnsupervisedValidation.read_result(test_pd_dataframeA)
        
        MatrixUtils.check_input_matrix(area_table.values,"area_table(np array)","")
        row_name_result = ["max","accumulate","min","score"]
        col_result = [1.0, 0.5, 0.0, 0.5]
        
        self.assertEqual(area_table.shape, (4, test_pd_dataframeA.shape[1]))
        for i in range(area_table.shape[0]):
            self.assertEqual(area_table.index[i], row_name_result[i])
        
        for i in range(area_table.shape[1]):
            for j in range(area_table.shape[0]):
                self.assertAlmostEqual(area_table.iloc[j,i], col_result[j])
                   
    
        
    def test_plot_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(
            np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        UnsupervisedValidation.plot_result(
            test_pd_dataframeA, "unittest_unsupervised_plot_result.svg", 
            format = "svg")    
        
    def test_plot_single_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(
            np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        UnsupervisedValidation.plot_single_result(
            test_pd_dataframeA, 0, "unittest_unsupervised_plot_single_result.svg", 
            format = "svg")  
                    

if __name__ == '__main__':
    unittest.main()
