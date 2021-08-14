# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 14:41:30 2021

@author: Bigghost
"""

import unittest
from KnowledgeCaptureValidation import *
from Utils import *
from Unittest_Utils import *

class InvalidSelectedFeaturesTestCases:
    def get_invalid_type_message(argument_order):
        return ("The input selected features is not a list!\n" + 
                Utils.generate_additional_description(argument_order)
                )
    
    def get_invalid_element_type_message(argument_order):
        return ("The input selected features contains non-string element!\n" + 
                Utils.generate_additional_description(argument_order)
                )
    
    def get_duplicated_elements_message(argument_order):
        return("The input selected features contains duplicated elements!\n" + 
               Utils.generate_additional_description(argument_order))

    def get_invalid_features_message(argument_order):
        return ("One or more features not listed in data matrix!\n" + 
                Utils.generate_additional_description(argument_order))
    
    def get_all_features_selected_message(argument_order):
        return ("All features are selected!\n" + 
                Utils.generate_additional_description(argument_order))
    
    def prepare_invalid_selected_features_testcases(argument_order, data_matrix_pd, func):
        test_case_list = [None,
                              "Test",
                              [1,2,3],
                              ["1","1","1"],
                              ["HAHA","TEST",""],
                              data_matrix_pd.index.tolist()
                              ]
        
        expected_error_type = [TypeError,
                           TypeError,
                           TypeError,
                           ValueError,
                           ValueError,
                           ValueError
                           ]
        
        expected_output_list = [
            InvalidSelectedFeaturesTestCases.get_invalid_type_message(argument_order),
            InvalidSelectedFeaturesTestCases.get_invalid_type_message(argument_order),
            InvalidSelectedFeaturesTestCases.get_invalid_element_type_message(argument_order),
            InvalidSelectedFeaturesTestCases.get_duplicated_elements_message(argument_order),
            InvalidSelectedFeaturesTestCases.get_invalid_features_message(argument_order),
            InvalidSelectedFeaturesTestCases.get_all_features_selected_message(argument_order)]
        
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func 
        )
        
        return target_function
        
class InvalidCondPairTestCases:
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("Input condition pair is not a tuple!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_len_message(argument_order):
        return ("Input condition pair should be a tuple with two strings!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_element_type_message(argument_order):
        return ("Input condition pair should be a tuple with two strings!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_same_condition_message(argument_order):
        return ("Two input conditions are the same!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_cond_pair_testcases(argument_order, func):
        test_case_list = [ None,
                          "Test",
                         (1,1,3),
                         (1,2),
                         ("1","1"),
                         ]
        expected_error_type = [ TypeError,
                                TypeError,
                                ValueError,
                                TypeError,
                                ValueError
                              ]
        
        expected_output_list = [
            InvalidCondPairTestCases.get_invalid_type_message(argument_order),
            InvalidCondPairTestCases.get_invalid_type_message(argument_order),
            InvalidCondPairTestCases.get_invalid_len_message(argument_order),
            InvalidCondPairTestCases.get_invalid_element_type_message(argument_order),
            InvalidCondPairTestCases.get_same_condition_message(argument_order)]
        
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func
            )
        
        return target_function
    
class InvalidInputConditionTestCases:
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("Input condition is not a pandas dataframe!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_header_message(argument_order):
        return ("The condition table should have " + 
                "at least one column with label 'Condition'" + 
                " which labeled the condition of samples.\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_element_type_message(argument_order):
        return ("The condition label should be a string!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_conditions_message(argument_order):
        return ("The condition table does not have the condition specified in given condition pair\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_samples_message(argument_order):
        return ("One or more samples listed in condition" + 
                " table do not existed in input data matrix!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_input_conditions_testcases(argument_order, func):
        test_case_list = [ None,
                          "Test",
                         pd.DataFrame(columns=["Test","Test1"]),
                         pd.DataFrame([1,1,2,2], columns = ["Condition"]),
                         pd.DataFrame(["1","1","2","2"], columns = ["Condition"]),
                         pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Test1","Test2","Test3","Test4"]),
                         ]
        expected_error_type = [ TypeError,
                                TypeError,
                                ValueError,
                                TypeError,
                                ValueError,
                                ValueError
                              ]
        
        expected_output_list = [
            InvalidInputConditionTestCases.get_invalid_type_message(argument_order),
            InvalidInputConditionTestCases.get_invalid_type_message(argument_order),
            InvalidInputConditionTestCases.get_invalid_header_message(argument_order),
            InvalidInputConditionTestCases.get_invalid_element_type_message(argument_order),
            InvalidInputConditionTestCases.get_invalid_conditions_message(argument_order),
            InvalidInputConditionTestCases.get_invalid_samples_message(argument_order)]
        
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func
            )
        
        return target_function
    

class InvalidChangeTendencyTestCases:
    @staticmethod
    def get_invalid_type_message(argument_order):
        return ("Change tendency should be a string with value 'up' or 'down'!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def get_invalid_option_message(argument_order):
        return ("Change tendency should be a string with value 'up' or 'down'!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def prepare_invalid_change_tendency_testcases(argument_order, func):
        test_case_list = [None,
                          1,
                          "Test"]
        expected_error_type = [TypeError,
                               TypeError,
                               ValueError]
        expected_output_list = [
            InvalidChangeTendencyTestCases.get_invalid_type_message(argument_order),
            InvalidChangeTendencyTestCases.get_invalid_type_message(argument_order),
            InvalidChangeTendencyTestCases.get_invalid_option_message(argument_order)]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            func
            )
        
        return target_function
    
class ValidateKnowledgeCaptureValidationResult:
    def validate_raw_result(pvalue, 
                            log_fold_change_pd, average_condA_pd, average_condB_pd):
        Utils.check_float(pvalue, "", "")
        Utils.check_value_boundary(pvalue < 0 and pvalue > 1, "", "") 
        
        test_list = [log_fold_change_pd, average_condA_pd, average_condB_pd]
        for x in test_list:
            for y in x :
                Utils.check_float(y, "", "")
        
    

class TestKnowledgeCaptureValidationModule(Test_Func_Base):
    
    def test_knowledge_capture_evaluation_one_pair_first_argument(self):
        target_function = InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases_typeonly(
            "Data matrix of condition A", "first argument",
            lambda x : KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                x, 
                None, 
                None, 
                None))
        self.base_test_matrix(target_function)
        
    def test_knowledge_capture_evaluation_one_pair_second_argument(self):
        target_function = InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases_typeonly(
            "Data matrix of condition B", "second argument",
            lambda x : KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                pd.DataFrame(1.0, columns=["S1","S2"], 
                             index = [("Feature" + str(y)) for y in range(20)]), 
                x, 
                None, 
                None))
        self.base_test_matrix(target_function)
    
    def test_knowledge_capture_evaluation_one_pair_inconsistent_features(self):
        test_case_list = [pd.DataFrame(1.0, columns=["S1","S2"], 
                             index = [("Feature" + str(x)) for x in range(20)])]
        expected_error_type = [ValueError]
        expected_output_list = ["One or more samples appear in data matrix of both conditions!\n" + 
                                 Utils.generate_additional_description("first and second argument")
                                ]
        target_function = TargetFunction(
            "",
            "",
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                x, 
                pd.DataFrame(1.0, columns=["S1","S4"], 
                             index = [("Feature" + str(y)) for y in range(20)]), 
                None, 
                None)
            )
        self.base_test_matrix(target_function)
        
    def test_knowledge_capture_evaluation_one_pair_inconsistent_features(self):
        test_case_list = [pd.DataFrame(1.0, columns=["S1","S2"], 
                             index = [("XFeature" + str(x)) for x in range(20)])]
        expected_error_type = [ValueError]
        expected_output_list = [("The features listed in data matrix of condition A is not equal to" + 
                                " the features listed in data matrix of condition B!\n" + 
                                Utils.generate_additional_description("first and second argument"))
                                ]
        target_function = TargetFunction(
            "",
            "",
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                x, 
                pd.DataFrame(1.0, columns=["S3","S4"], 
                             index = [("Feature" + str(y)) for y in range(20)]), 
                None, 
                None)
            )
        self.base_test_matrix(target_function)
        
    def test_knowledge_capture_evaluation_one_pair_third_argument(self):
        condA_mixed_matrix_pd = pd.DataFrame(1.0, columns=["S1","S2"], index = [("Feature" + str(y)) for y in range(20)])
        condB_mixed_matrix_pd = pd.DataFrame(2.0, columns=["S3","S4"], index = [("Feature" + str(y)) for y in range(20)])
       
        target_function = \
            InvalidSelectedFeaturesTestCases.prepare_invalid_selected_features_testcases(
                "third argument", 
                
                pd.DataFrame(1.0, columns=["S1","S2"], 
                 index = [("Feature" + str(y)) for y in range(20)]), 
                
                lambda x : KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                    condA_mixed_matrix_pd, 
                    condB_mixed_matrix_pd, 
                    x, 
                    None)
                )
            
        self.base_test_matrix(target_function)
        
    def test_knowledge_capture_evaluation_one_pair_fourth_argument(self):
        condA_mixed_matrix_pd = pd.DataFrame(1.0, columns=["S1","S2"], index = [("Feature" + str(y)) for y in range(20)])
        condB_mixed_matrix_pd = pd.DataFrame(2.0, columns=["S3","S4"], index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidChangeTendencyTestCases.prepare_invalid_change_tendency_testcases(
                "fourth argument",
                lambda x : KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                    condA_mixed_matrix_pd, 
                    condB_mixed_matrix_pd, 
                    condA_mixed_matrix_pd.index.tolist()[0:5], 
                    x)
                )
        self.base_test_matrix(target_function)
        
    def test_knowledge_capture_evaluation_normal(self):
        condA_mixed_matrix_pd = pd.DataFrame(1.0, columns=["S1","S2"], index = [("Feature" + str(y)) for y in range(20)])
        condB_mixed_matrix_pd = pd.DataFrame(2.0, columns=["S3","S4"], index = [("Feature" + str(y)) for y in range(20)])
        
        for x in ["up","down"]:
            p_value, log_fold_change_pd, average_condA_pd, average_condB_pd = \
                KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
                        condA_mixed_matrix_pd, 
                        condB_mixed_matrix_pd, 
                        condA_mixed_matrix_pd.index.tolist()[0:5],
                        x
                        )
            
            ValidateKnowledgeCaptureValidationResult.validate_raw_result(
                p_value, log_fold_change_pd, average_condA_pd, average_condB_pd)
            
    
    def test_validate_data_matrix_specific_noise_ratio_first_argument(self):
        target_function = InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases_typeonly(
            "input compendium", "first argument", 
            lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                x, 
                None, 
                None, 
                None, 
                None, 
                None, 
                None))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_second_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = InvalidMatrixTestCases.prepare_invalid_matrix_testcases(
            "noise matrix", "second argument",
            lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                good_matrix_pd, 
                x, 
                None, 
                None, 
                None, 
                None, 
                None))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_inconsistent_shape(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = InvalidMatrixTestCases.prepare_inconsistent_matrix_shape_testcases(
            "input data matrix", "noise matrix",
            "first and second argument",
            lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                good_matrix_pd, 
                x, 
                None, 
                None, 
                None, 
                None, 
                None))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_third_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidCondPairTestCases.prepare_invalid_cond_pair_testcases(
                "third argument",
                lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                good_matrix_pd, 
                np.random.rand(good_matrix_pd.shape[0],good_matrix_pd.shape[1]), 
                x, 
                None, 
                None, 
                None, 
                None))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_fourth_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidInputConditionTestCases.prepare_invalid_input_conditions_testcases(
                "fourth argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                good_matrix_pd, 
                np.random.rand(good_matrix_pd.shape[0],good_matrix_pd.shape[1]), 
                ("Condition 1","Condition 2"), 
                x, 
                None, 
                None, 
                None))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_fifth_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidSelectedFeaturesTestCases.prepare_invalid_selected_features_testcases(
                "fifth argument", 
                good_matrix_pd, 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                    good_matrix_pd, 
                    np.random.rand(good_matrix_pd.shape[0],good_matrix_pd.shape[1]), 
                    ("Condition 1","Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    x, 
                    None, 
                    None))
        self.base_test_matrix(target_function)
        
        
    def test_validate_data_matrix_specific_noise_ratio_6th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidChangeTendencyTestCases.prepare_invalid_change_tendency_testcases(
                "6th argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                    good_matrix_pd, 
                    np.random.rand(good_matrix_pd.shape[0],good_matrix_pd.shape[1]), 
                    ("Condition 1","Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5], 
                    x, 
                    None))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_7th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidNoiseRatioTestCases.prepare_invalid_noise_ratio_cases(
                "7th argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                    good_matrix_pd, 
                    np.random.rand(good_matrix_pd.shape[0],good_matrix_pd.shape[1]), 
                    ("Condition 1","Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5], 
                    "up", 
                    x))
            
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_specific_noise_ratio_normal(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        p_value, log_fold_change_selected, log_fold_change_others = \
            KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                    good_matrix_pd, 
                    np.random.rand(good_matrix_pd.shape[0],good_matrix_pd.shape[1]), 
                    ("Condition 1","Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5], 
                    "up", 
                    0.5)
            
        Utils.check_float(p_value)
        Utils.check_value_boundary(p_value < 0.0 or p_value > 1.0)
        
        for x in log_fold_change_selected:
            Utils.check_float(x)
        for x in log_fold_change_others:
            Utils.check_float(x)
            
    def test_validate_data_matrix_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases_typeonly(
                "input compendium", "first argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    x, 
                    None, 
                    None, 
                    None, 
                    None))
        
        self.base_test_matrix(target_function)
            
    def test_validate_data_matrix_second_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidCondPairTestCases.prepare_invalid_cond_pair_testcases(
                "second argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    x, 
                    None, 
                    None, 
                    None))  
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_third_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidInputConditionTestCases.prepare_invalid_input_conditions_testcases(
                "third argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    x, 
                    None, 
                    None))  
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_fourth_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidSelectedFeaturesTestCases.prepare_invalid_selected_features_testcases(
                "fourth argument", 
                good_matrix_pd,
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    x, 
                    None))  
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_fifth_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidChangeTendencyTestCases.prepare_invalid_change_tendency_testcases(
                "fifth argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    x))  
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_6th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidNoiseRatioStepTestCases.prepare_invalid_noise_ratio_step_cases(
                "6th argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    x))  
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_7th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        target_function = \
            InvalidNTrialTestCases.prepare_invalid_n_trial_cases(
                "7th argument", 
                lambda x : KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    x))  
        
        self.base_test_matrix(target_function)
        
    def test_validate_data_matrix_normal(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        
        ResultUtils.check_general_result(result, "", "", lower_limit = 0.0, upper_limit = 1.0)
        ResultUtils.check_general_result(avg_selected_log_fc, "", "")
        ResultUtils.check_general_result(avg_others_log_fc, "", "")
        ResultUtils.check_general_result(std_selected_log_fc, "", "")
        ResultUtils.check_general_result(std_others_log_fc, "", "")
        
        Utils.check_int(n_selected)
        Utils.check_int(n_others)
        
    def test_plot_result(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "knowledge capture validation result", 
                "first argument", 
                lambda x : KnowledgeCaptureValidation.plot_result(x, "Test.svg", format="svg"), 
                lower_limit = 0.0, upper_limit = 1.0)
        self.base_test_matrix(target_function)
            
    def test_plot_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        KnowledgeCaptureValidation.plot_result(test_pd_dataframeA, "Test.svg", format="svg")
        
    def test_parse_result_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "knowledge capture validation result", 
                "first argument", 
                lambda x : KnowledgeCaptureValidation.parse_result(x, None),
                lower_limit = 0.0, upper_limit = 1.0)
        self.base_test_matrix(target_function)
                
    def test_parse_result_second_argument(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        target_function = \
            InvalidPValueThresholdTestCases.prepare_invalid_pvalue_threshold_cases(
                "second argument", 
                lambda x : KnowledgeCaptureValidation.parse_result(
                    test_pd_dataframeA, x))
        self.base_test_matrix(target_function)
            
    def test_parse_result_normal(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        scores = KnowledgeCaptureValidation.parse_result(test_pd_dataframeA, 0.05)
        ResultUtils.check_general_score(scores, "", "")
        
    
    def test_plot_avg_log_fold_changes_first_argument(self):
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "knowledge capture validation result", 
                "first argument", 
                lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(x, 
                  None, None, None, None, None,\
                  None, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg'),
                lower_limit = 0.0, upper_limit = 1.0)
        self.base_test_matrix(target_function)
                
    def test_plot_avg_fold_changes_second_argument(self):
        test_pd_dataframeA = pd.DataFrame(np.array([np.arange(0,1.1,0.1)]*10).transpose(), index = np.arange(0,1.1,0.1))
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_scores_testcases(
                "knowledge capture validation score", 
                "second argument", 
                lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    test_pd_dataframeA, 
                    x, 
                    None, None, None, None,\
                    None, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg'))
        self.base_test_matrix(target_function)
                
    def test_plot_avg_fold_changes_third_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05)

        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "average fold change (selected features)", 
                "third argument", 
                lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    x, None, None, None,\
                    None, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg'))
        self.base_test_matrix(target_function)
                
    def test_plot_avg_fold_changes_fourth_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "average fold change (other features)", 
                "fourth argument", 
                lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    avg_selected_log_fc, x, None, None,\
                    None, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg'))
        self.base_test_matrix(target_function)
    
    def test_plot_avg_fold_changes_fifth_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "std. of fold change (selected features)", 
                "fifth argument", 
                lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    avg_selected_log_fc, avg_others_log_fc, x, None,\
                    None, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg'))
        self.base_test_matrix(target_function)
        
    def test_plot_avg_fold_changes_6th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05)
        
        target_function = \
            InvalidMatrixTestCases.prepare_invalid_pd_dataframe_testcases(
                "std. of fold change (other features)", 
                "6th argument", 
                lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, x,\
                    None, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg'))
        self.base_test_matrix(target_function)
        
    def test_plot_avg_fold_changes_7th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05)
        
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
            "#selected features should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#selected features should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#selected features should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#selected features should be positive!" + 
            Utils.generate_additional_description(argument_order),
            
            "#selected features should be positive!" + 
            Utils.generate_additional_description(argument_order),
        ]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc,\
                    x, None, '0', 'knowledge_capture_avg_fold_change.svg', format='svg')
            )
        self.base_test_matrix(target_function)
        
    def test_plot_avg_fold_changes_8th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05) 
        
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
            "#other features should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#other features should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#other features should be an integer!" + 
            Utils.generate_additional_description(argument_order),
            
            "#other features should be positive!" + 
            Utils.generate_additional_description(argument_order),
            
            "#other features should be positive!" + 
            Utils.generate_additional_description(argument_order),
        ]
        
        target_function = TargetFunction(
            "",
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc,\
                    n_selected, x, '0', 'knowledge_capture_avg_fold_change.svg', format='svg')
            )
        self.base_test_matrix(target_function)
        
    
    def test_plot_avg_fold_changes_9th_argument(self):
        good_matrix_pd = pd.DataFrame(np.random.rand(20,20), 
                                      columns=[("Sample" + str(y)) for y in range(20)], 
                                      index = [("Feature" + str(y)) for y in range(20)])
        
        result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
            n_selected, n_others = \
        KnowledgeCaptureValidation.validate_data_matrix(
                    good_matrix_pd, 
                    ("Condition 1", "Condition 2"), 
                    pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                      index = ["Sample1","Sample2","Sample3","Sample4"]),
                    good_matrix_pd.index.tolist()[0:5],  
                    "up",
                    0.25,
                    3)
        scores = KnowledgeCaptureValidation.parse_result(result, 0.05) 
        
        target_function = InvalidNTrialTestCases.prepare_invalid_trial_index_cases(
            "9th argument", 
            lambda x : KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                    result, 
                    scores, 
                    avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
                    n_selected, n_others, x, 'knowledge_capture_avg_fold_change.svg', format='svg'))
            
        self.base_test_matrix(target_function)
    
    def test_plot_avg_fold_changes_normal(self):
        shape_list = [20, 50, 100]
        noise_ratio_step_list = [0.25, 0.5]
        for shape in shape_list:
            for noise_ratio_step in noise_ratio_step_list:
                for n_trial in [1,3,5]:
                    good_matrix_pd = pd.DataFrame(np.random.rand(shape,shape), 
                                      columns=[("Sample" + str(y)) for y in range(shape)], 
                                      index = [("Feature" + str(y)) for y in range(shape)])
        
                    result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
                        n_selected, n_others = \
                    KnowledgeCaptureValidation.validate_data_matrix(
                                good_matrix_pd, 
                                ("Condition 1", "Condition 2"), 
                                pd.DataFrame(["Condition 1","Condition 1","Condition 2","Condition 2"], columns=["Condition"],
                                                  index = ["Sample1","Sample2","Sample3","Sample4"]),
                                good_matrix_pd.index.tolist()[0:5],  
                                "up",
                                noise_ratio_step,
                                n_trial)
                    scores = KnowledgeCaptureValidation.parse_result(result, 0.05) 
                    
                    KnowledgeCaptureValidation.plot_avg_log_fold_changes(
                                result, 
                                scores, 
                                avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
                                n_selected, n_others, n_trial - 1, 'knowledge_capture_avg_fold_change.svg', format='svg')
        
       
        
        
    
if __name__ == '__main__':
    unittest.main()
