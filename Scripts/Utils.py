# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:56:09 2021

@author: Bigghost
"""

import numpy as np
import pandas as pd
from scipy import stats

class MatrixUtils:
    @staticmethod
    def check_input_matrix(data_matrix, 
                           input_matrix_name = "input matrix", 
                           argument_order_description = "first argument",
                           lower_limit = None,
                           upper_limit = None):
        
        if type(input_matrix_name) is not str:
            raise TypeError(
                "The argument of input checking method is not a string!\n" +
                "Check the second argument if you call this method directly,\n" + 
                "otherwise please contact the author.")
        if type(argument_order_description) is not str:
            raise TypeError(
                "The argument of input checking method is not a string!\n" +
                "Check the third argument if you call this method directly,\n" + 
                "otherwise please contact the author.")
            
        
        additional_description = \
            Utils.generate_additional_description(argument_order_description)
        
            
        if type(data_matrix) is not np.ndarray:
            raise TypeError(
                input_matrix_name.capitalize() + 
                " is not a numpy n-dimension array!\n" + 
                additional_description)
        if len(data_matrix.shape) != 2:
            raise ValueError(
                "The dimension of " + input_matrix_name + " should be 2D!\n" + 
                additional_description)
        if data_matrix.shape[0] == 0:
            raise ValueError(
                "#Row of " + input_matrix_name + " is zero!\n" + 
                additional_description)
        if data_matrix.shape[1] == 0:
            raise ValueError(
                "#Column of " + input_matrix_name + " is zero!\n" + 
                additional_description)
        try:
            if np.isnan(np.sum(data_matrix)):
                raise ValueError(
                    "NaN values detected in " + input_matrix_name + "!\n" + 
                    additional_description)
            if np.isinf(np.sum(data_matrix)):
                raise ValueError(
                    "Infinite values detected in " + input_matrix_name + "!\n" + 
                    additional_description)
        except TypeError:
            raise TypeError(
                    "Non-numerical values detected in " + input_matrix_name + "!\n" + 
                    additional_description)
            
        
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                if lower_limit is not None:
                    if data_matrix[i][j] < lower_limit:
                        raise ValueError(
                            "One or more values less than lower_limit (= " + 
                            str(lower_limit) + ") in " + input_matrix_name + "!\n" + 
                        additional_description)
                    
                if upper_limit is not None:
                    if data_matrix[i][j] > upper_limit:
                        raise ValueError(
                            "One or more values higher than upper_limit (= " + 
                            str(upper_limit) + ") in " + input_matrix_name + "!\n" + 
                        additional_description)
            
    @staticmethod
    def check_two_matrix_shape_consistent(matrix_A, matrix_B, 
                                          name_A, name_B, argument_order):
        additional_description = \
            Utils.generate_additional_description(argument_order)
        
        if matrix_A.shape != matrix_B.shape:
            raise ValueError("The sizes of " + name_A + 
                             " and " + name_B + " are different!\n" + 
                Utils.generate_additional_description(argument_order))
            
    @staticmethod
    def check_input_pd_dataframe(pd_dataframe, pd_dataframe_name, argument_order, 
                             lower_limit = None, upper_limit = None):
        if type(pd_dataframe) is not pd.core.frame.DataFrame:
            raise TypeError("The " + pd_dataframe_name + " is not a Pandas Dataframe!\n" + 
                Utils.generate_additional_description(argument_order))
            
        MatrixUtils.check_input_matrix(pd_dataframe.values, 
                                       pd_dataframe_name, argument_order,
                                       lower_limit, upper_limit)
                
class NoiseRatioUtils:
    @staticmethod
    def check_noise_ratio(noise_ratio, argument_order):
        Utils.check_float(noise_ratio, 
                          "The noise ratio is not a float!\n", 
                          argument_order)
        
        Utils.check_value_boundary(noise_ratio < 0,
                                   "The noise ratio is too low: \n" + 
                "The noise ratio should not be lower than 0!\n",
                                    argument_order)
            
        Utils.check_value_boundary(noise_ratio > 1,
                                   "The noise ratio is too high: \n" + 
                "The noise ratio should not be higher than 1!\n",
                                    argument_order)
        
                
    @staticmethod
    def check_noise_ratio_step(noise_ratio_step, argument_order):
        Utils.check_float(noise_ratio_step, 
                          "The noise ratio step is not a float!\n", 
                          argument_order)
        Utils.check_value_boundary(noise_ratio_step <= 0, 
                                   "The noise ratio step is too low: \n" + 
                "The noise ratio step should be greater than 0!\n", 
                                   argument_order)
        Utils.check_value_boundary(noise_ratio_step > 0.5, 
                                   "The noise ratio step is too high: \n" + 
                "The noise ratio step should not be higher than 0.5!\n", 
                                   argument_order)
        
class PValueUtils:
    @staticmethod
    def check_pvalue_threshold(pvalue_threshold, argument_order):
        Utils.check_float(pvalue_threshold, 
                          "The p-value threshold is not a float!\n", 
                          argument_order)
        
        Utils.check_value_boundary(pvalue_threshold < 0,
                                   "The p-value threshold is too low: \n" + 
                "The p-value threshold should not be lower than 0!\n",
                                    argument_order)
            
        Utils.check_value_boundary(pvalue_threshold > 1,
                                   "The p-value threshold is too high: \n" + 
                "The p-value threshold should not be higher than 1!\n",
                                    argument_order)
        
                
class NTrialUtils:
    @staticmethod
    def check_n_trial(n_trial, argument_order):
        Utils.check_int(n_trial, 
                        "The number of trial is not an integer!\n", 
                        argument_order)
        
        Utils.check_value_boundary(n_trial <= 0,
                                   "The number of trial should be greater than zero!\n", 
                                   argument_order)
            
    @staticmethod
    def check_trial_index(trial_index, input_result, argument_order):
        Utils.check_int(trial_index, 
                        "The trial index is not an integer!\n", argument_order)
        Utils.check_value_boundary(trial_index < 0,
                                   "The trial index should not be negative!\n",
                                   argument_order)
        Utils.check_value_boundary(trial_index >= input_result.shape[1],
                                   "The trial index should not be greater than or equal to #column of input result!\n",
                                   argument_order)
        
class ResultUtils:
    @staticmethod
    def check_general_result(result, result_name, argument_order, 
                             lower_limit = None, upper_limit = None):
        
        MatrixUtils.check_input_pd_dataframe(result, 
                                             result_name, 
                                             argument_order,
                                             lower_limit,
                                             upper_limit)
        
        for noise_ratio in result.index.values:
            NoiseRatioUtils.check_noise_ratio(noise_ratio, argument_order)
            
    
    @staticmethod
    def check_unsupervised_validation_result(result, argument_order):
        ResultUtils.check_general_result(result, "unsupervised validation result", argument_order,
                                   lower_limit = 0.0) #Error should not be < 0
        
    
    @staticmethod
    def check_correlation_validation_result(result, argument_order):
        #pvalue should be between 0 to 1
        ResultUtils.check_general_result(result, "correlation validation result", argument_order,
                                   lower_limit = 0.0, upper_limit = 1.0)
        
    @staticmethod
    def check_knowledge_capture_validation_result(result, argument_order):
        #pvalue should be between 0 to 1
        ResultUtils.check_general_result(result, "knowledge capture validation result", argument_order,
                                   lower_limit = 0.0, upper_limit = 1.0)
        
    
    @staticmethod
    def check_general_score(scores, score_name, argument_order):
        #score should be between 0 to 1
        if type(scores) is not pd.core.frame.DataFrame:
            raise TypeError("The " + score_name + " is not a Pandas Dataframe!\n" + 
                Utils.generate_additional_description(argument_order))
        
        if len(scores.columns) != 1:
            raise ValueError("The " + score_name + " should have only one column with name 'Score'!\n" + 
                Utils.generate_additional_description(argument_order))
        if scores.columns[0] != 'Score':
            raise ValueError("The " + score_name + " should have only one column with name 'Score'!\n" + 
                Utils.generate_additional_description(argument_order))
            
        MatrixUtils.check_input_matrix(scores.values, score_name, argument_order,
                                       lower_limit = 0.0, upper_limit = 1.0)
        

class Utils:
    @staticmethod
    def check_float(input_val, message = "", argument_order = ""):
        if type(input_val) is not float and \
            not np.issubdtype(type(input_val), np.floating):
                raise TypeError(
                message + Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def check_int(input_val, message = "", argument_order = ""):
        if type(input_val) is not int and \
            not np.issubdtype(type(input_val),np.integer):
                raise TypeError(
                message + Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def check_value_boundary(cond, message = "", argument_order = ""):
        if cond:
            raise ValueError(
                message + Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def generate_additional_description(
            argument_order_description = "first argument"):
        
        if type(argument_order_description) is not str:
            raise TypeError(
                "The argument of input checking method is not a string!\n" +
                "Check the argument if you call this method directly,\n" + 
                "otherwise please contact the author.")
            
        if argument_order_description == "":
            return ""
        
        additional_description = \
                "Check the " + argument_order_description + \
                " if you call this method directly,\n" + \
                "otherwise please contact the author."
                
        return additional_description
        
    @staticmethod
    def prepare_noise(data_matrix):
        MatrixUtils.check_input_matrix(data_matrix)
        
        noise = np.copy(data_matrix)
        noise = noise.reshape((data_matrix.shape[0]*data_matrix.shape[1],1))
        noise = np.random.permutation(noise)
        noise = noise.reshape(data_matrix.shape)
        return noise
        
        