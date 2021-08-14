# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:28:51 2021

@author: Bigghost
"""

import unittest
from PublishedDataComparison import *
from Utils import *
from Unittest_Utils import *

class TestPublishedDataComparisonModule(Test_Func_Base):
    def test_data_comparison_first_argument(self):
        argument_name = "input compendium"
        argument_order = "first argument"
        
        test_case_list = [ None,
                          "Test",
                          pd.DataFrame(1,columns=["1","2","3"],index=["1","1","2"]),
                          pd.DataFrame(1,columns=["1","1","2"],index=["1","2","3"]),
                          ]
        
        expected_error_type = [ TypeError,
                                TypeError,
                                ValueError,
                                ValueError]
        
        expected_output_list = [
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        
        ("Input compendium contains duplicated features!\n" + 
         Utils.generate_additional_description(argument_order)),
        ("Input compendium contains duplicated samples!\n" + 
         Utils.generate_additional_description(argument_order))
        
        ]
        
        target_function = TargetFunction(
            argument_name,
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : PublishedDataComparison.data_comparison(x, 
                        None, 
                        None,
                        xlabel = None,
                        ylabel = None,
                        title = None))
        
        self.base_test_matrix(target_function)
        
    def test_data_comparison_second_argument(self):
        argument_name = "published data"
        argument_order = "second argument"
        
        test_case_list = [ None,
                          "Test",
                          pd.DataFrame(1,columns=["1","2","3"],index=["1","1","2"]),
                          pd.DataFrame(1,columns=["1","1","2"],index=["1","2","3"]),
                          ]
        
        expected_error_type = [ TypeError,
                                TypeError,
                                ValueError,
                                ValueError]
        
        expected_output_list = [
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        InvalidMatrixTestCases.get_invalid_type_pd_array_message(
            argument_name, argument_order),
        
        ("Published data contains duplicated features!\n" + 
         Utils.generate_additional_description(argument_order)),
        ("Published data contains duplicated samples!\n" + 
         Utils.generate_additional_description(argument_order))
        
        ]
        
        good_dataset = pd.DataFrame(1,columns=["1","2","3"],index=["1","2","3"])
        
        target_function = TargetFunction(
            argument_name,
            argument_order,
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : PublishedDataComparison.data_comparison(good_dataset, 
                        x, 
                        None,
                        xlabel = None,
                        ylabel = None,
                        title = None))
        
        self.base_test_matrix(target_function)
        
    def test_data_comparison_no_common_features(self):
        
        test_case_list = [pd.DataFrame(1,columns=["1","2","3"],index=["4","5","6"]),
                          pd.DataFrame(1,columns=["1","2","3"],index=["1","5","6"])]
        
        expected_error_type = [ValueError,
                               ValueError]
        
        expected_output_list = [("Too few Common features found ( < 2) in input compendium and published data!\n" +
                                 Utils.generate_additional_description("first and second argument")),
                                ("Too few Common features found ( < 2) in input compendium and published data!\n" +
                                 Utils.generate_additional_description("first and second argument"))]
        
        good_dataset = pd.DataFrame(1,columns=["1","2","3"],index=["1","2","3"])
        
        target_function = TargetFunction(
            "",
            "",
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : PublishedDataComparison.data_comparison(good_dataset, 
                        x, 
                        None,
                        xlabel = None,
                        ylabel = None,
                        title = None))
        
        self.base_test_matrix(target_function)
        
    def test_data_comparison_no_common_samples(self):
        good_dataset = pd.DataFrame(1,columns=["1","2","3"],index=["1","2","3"])
        
        test_case_list = [pd.DataFrame(1,columns=["1","2","4"],index=["1","2"]),
                          pd.DataFrame(1,columns=["1","3|5"],index=["1","2"]),]
        
        expected_error_type = [ValueError,
                               ValueError]
        
        expected_output_list = [("The sample " + "4 " + "is not found in input compendium.\n" + 
                                 Utils.generate_additional_description("first argument")),
                                ("The sample " + "5 " + "is not found in input compendium.\n" + 
                                 Utils.generate_additional_description("first argument"))]
        
        
        target_function = TargetFunction(
            "",
            "",
            test_case_list,
            expected_error_type,
            expected_output_list,
            lambda x : PublishedDataComparison.data_comparison(good_dataset, 
                        x, 
                        None,
                        xlabel = None,
                        ylabel = None,
                        title = None))
        
        self.base_test_matrix(target_function)
        
    def test_data_comparison_normal(self):
        good_dataset = pd.DataFrame(np.random.rand(20,20),columns=[str(x) for x in range(20)],index=[str(x) for x in range(20)])
        
        test_case_list = [pd.DataFrame(good_dataset.values[0:10,0:10] * 0.5 + np.random.rand(10,10)* 0.5, 
                                      columns = [str(x) for x in range(10)],
                                      index = [str(x) for x in range(10)])]
        
        for test_case in test_case_list:
            corr, data_matrix_for_comparison_pd, published_data_matrix_pd = \
                PublishedDataComparison.data_comparison(good_dataset, 
                            test_case, 
                            "test_published_data_comparison.png",
                            xlabel = "",
                            ylabel = "",
                            title = "",
                            format ="png")
     
            self.assertEqual(corr.shape, (data_matrix_for_comparison_pd.shape[1],1))
            self.assertEqual(corr.columns[0], "corr")
            self.assertEqual(data_matrix_for_comparison_pd.shape, published_data_matrix_pd.shape)
            self.assertEqual(set(data_matrix_for_comparison_pd.index), set(published_data_matrix_pd.index))
            self.assertEqual(set(data_matrix_for_comparison_pd.columns), set(published_data_matrix_pd.columns))
        
if __name__ == '__main__':
    unittest.main()