# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:52:48 2021

KnowledgeCaptureValidation.py: 
Evaluate the quality of an omics compendium given two groups of samples and
features which are expected to be significantly increase/decrease between
two groups.

This approach assumes a high quality compendium should be capable to capture
the information including the difference of the features of two groups. If 
there are specific features (i.e. gene expressions) will be significantly
increase/decrease according to the published data, the compendium should be
capable to keep this kind of information. The significance can be verified by
applying one-tailed Wilcoxon ranksum test of the fold change of selected
features and the fold change of non-selected features.

As the noise ratio increase, the assumption may become negative. The highest
noise ratio which still makes the assumption true can be defined as the quality
of the compendium.


@author: ChengEn Tan (cetan@ucdavis.edu)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from Utils import *

import argparse

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class SelectedFeaturesUtils:
    '''
    Utilities for checking the input selected features
    
    Methods in this class will be called internally for checking the input
    '''
    
    @staticmethod
    def check(selected_features, data_matrix_pd, argument_order):
        SelectedFeaturesUtils.check_type(selected_features, argument_order)
        SelectedFeaturesUtils.check_element_type(selected_features, argument_order)
        SelectedFeaturesUtils.check_duplicate_features(selected_features, argument_order)
        SelectedFeaturesUtils.check_valid_features(
            selected_features, data_matrix_pd, argument_order)
        SelectedFeaturesUtils.check_all_features_selected(
            selected_features, data_matrix_pd, argument_order)
    
    @staticmethod
    def check_type(selected_features, argument_order):
        if type(selected_features) is not list:
            raise TypeError("The input selected features is not a list!\n" + 
                            Utils.generate_additional_description(argument_order)
                            )
            
    @staticmethod
    def check_element_type(selected_features, argument_order):
        for x in selected_features:
            if type(x) is not str and type(x) is not np.str and type(x) is not np.str_:
                raise TypeError("The input selected features contains non-string element!\n" + 
                                Utils.generate_additional_description(argument_order)
                                )
                
    @staticmethod
    def check_duplicate_features(selected_features, argument_order):
        if len(set(selected_features)) != len(selected_features):
            raise ValueError("The input selected features contains duplicated elements!\n" + 
                            Utils.generate_additional_description(argument_order)
                            )
                
    @staticmethod
    def check_valid_features(selected_features, data_matrix_pd, argument_order):
        for x in selected_features:
            if x not in data_matrix_pd.index:
                raise ValueError("One or more features not listed in data matrix!\n" + 
                                 Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def check_all_features_selected(selected_features, data_matrix_pd, argument_order):            
        if len(set(selected_features)) == len(set(data_matrix_pd.index)):
            raise ValueError("All features are selected!\n" + 
                                 Utils.generate_additional_description(argument_order))
                
class CondPairUtils:
    '''
    Utilities for checking the input condition pairs
    
    Methods in this class will be called internally for checking the input
    '''
    
    @staticmethod
    def check_cond_pair(cond_pair, argument_order):
        if type(cond_pair) is not tuple:
            raise TypeError("Input condition pair is not a tuple!\n" + 
                            Utils.generate_additional_description(argument_order))
            
        if len(cond_pair) != 2:
            raise ValueError("Input condition pair should be a tuple with two strings!\n" + 
                            Utils.generate_additional_description(argument_order))
              
        for x in cond_pair:
            if type(x) is not str and type(x) is not np.str and type(x) is not np.str_:
                raise TypeError("Input condition pair should be a tuple with two strings!\n" + 
                            Utils.generate_additional_description(argument_order))
            
        if cond_pair[0] == cond_pair[1]:
            raise ValueError("Two input conditions are the same!\n" + 
                             Utils.generate_additional_description(argument_order))
             
class InputConditionUtils():
    '''
    Utilities for checking the input condition table
    
    Methods in this class will be called internally for checking the input
    '''
    
    @staticmethod
    def check(conditions, cond_pair, data_matrix_pd, argument_order):
        InputConditionUtils.check_type(conditions, argument_order)
        InputConditionUtils.check_column(conditions, argument_order)
        InputConditionUtils.check_element_type(conditions, argument_order)
        InputConditionUtils.check_conditions(conditions, cond_pair, argument_order)
        InputConditionUtils.check_samples(conditions, data_matrix_pd, argument_order)
    
    @staticmethod
    def check_type(conditions, argument_order):
        if type(conditions) is not pd.core.frame.DataFrame:
            raise TypeError("Input condition is not a pandas dataframe!\n" + 
                           Utils.generate_additional_description(argument_order))
           
    @staticmethod
    def check_column(conditions, argument_order):
        if 'Condition' not in conditions.columns:
            raise ValueError("The condition table should have " + 
                            "at least one column with label 'Condition'" + 
                            " which labeled the condition of samples.\n" + 
                            Utils.generate_additional_description(argument_order))
            
    @staticmethod
    def check_element_type(conditions, argument_order):
        for x in conditions.loc[:,'Condition']:
            if type(x) is not str and type(x) is not np.str and type(x) is not np.str_:
                raise TypeError("The condition label should be a string!\n" + 
                            Utils.generate_additional_description(argument_order))
           
    @staticmethod
    def check_conditions(conditions, cond_pair, argument_order):
        for x in set(cond_pair):
            if x not in set(conditions.loc[:,'Condition']):
                raise ValueError("The condition table does not have the condition specified in given condition pair\n" + 
                                 Utils.generate_additional_description(argument_order))
            
    @staticmethod
    def check_samples(conditions, data_matrix_pd, argument_order):
        for x in conditions.index:
            if x not in data_matrix_pd.columns:
                raise ValueError("One or more samples listed in condition" + 
                                 " table do not existed in input data matrix!\n" + 
                                 Utils.generate_additional_description(argument_order))
                
class ChangeTendencyUtils:
    '''
    Utilities for checking the input change tendency
    
    Methods in this class will be called internally for checking the input
    '''
    
    @staticmethod
    def check(change_tendency, argument_order):
        if type(change_tendency) is not str and \
            type(change_tendency) is not np.str and \
                type(change_tendency) is not np.str_:
            raise TypeError("Change tendency should be a string with value 'up' or 'down'!\n" + 
                            Utils.generate_additional_description(argument_order))
            
        if change_tendency != "up" and change_tendency != "down":
            raise ValueError("Change tendency should be a string with value 'up' or 'down'!\n" + 
                            Utils.generate_additional_description(argument_order))
                    
    
class KnowledgeCaptureValidation:
    
    @staticmethod
    def knowledge_capture_evaluation_one_pair(condA_mixed_matrix_pd,
                                              condB_mixed_matrix_pd,
                                              selected_features, 
                                              change_tendency):  
        """
        Evaluate the (log2) fold change of features between two conditions.
        The fold change will be evaluated by the following formula: 
            log2((1 + average_values(condB))/(1 + average_values(condA)))
            
        After the (log2) fold change of features are evaluated, the fold change
        of selected features will be compared with the fold change of other
        features by performing one-tailed Wilcoxon ranksum test with specified
        expected change tendency.
        
        up : Higher values are expected in condition B compared with condition A
        down : Lower values are expected in condition B compared with condtion A

        Parameters
        ----------
        condA_mixed_matrix_pd : A pandas DataFrame
            Data matrix of condition A. 
            Generally it is expected to be a perturbed matrix (a linear 
            combination of original matrix and noise with different noise ratio)
            Columns represent samples, and rows represent features.
            
            A dataframe with only one sample is allowed. 
            
        condB_mixed_matrix_pd : A pandas DataFrame
            Data matrix of condition A. 
            Generally it is expected to be a perturbed matrix (a linear 
            combination of original matrix and noise with different noise ratio)
            Columns represent samples, and rows represent features.
            
            A dataframe with only one sample is allowed. 
            
            The features listed in data matrix of condition A and condition B
            should be the same with the same order.
            No samples are allowed to be listed in both condition A and
            condition B.
            
        selected_features : A list of string
            Features which is expected to be significantly increase/decrease
            in condition B compared with condition A.
            
            All features should be existed in the dataframe of condition A and 
            B. In addition, no duplicated features are allowed and it is not
            allowed to select all features listed in the dataframe of condition
            A (and B).
            
        change_tendency : A string. Only "up" and "down" is allowed
            The change tendency will be used in one-tailed Wilcoxon ranksum 
            test for comparing the fold change of selected features and other 
            features.


        Returns
        -------
        p_value : A float
            The p-value of one-tailed Wilcoxon ranksum test of comparison
            between the fold change of selected features and the fold change
            of other features.
            
        log_fold_change_pd : A pandas series
            The log2 fold change of all features listed in data matrix.
            
        average_condA_pd : A pandas series
            The average values of all features among samples from condition A.
            
        average_condB_pd : A pandas series
            The average values of all features among samples from condition B.

        """
        
        MatrixUtils.check_input_pd_dataframe(
            condA_mixed_matrix_pd, "Data matrix of condition A", "first argument")
        
        MatrixUtils.check_input_pd_dataframe(
            condB_mixed_matrix_pd, "Data matrix of condition B", "second argument")
 
        for x in condA_mixed_matrix_pd.columns:
            if x in condB_mixed_matrix_pd.columns:
                raise ValueError("One or more samples appear in data matrix of both conditions!\n" + 
                                 Utils.generate_additional_description("first and second argument"))

        
        if set(condA_mixed_matrix_pd.index) != set(condB_mixed_matrix_pd.index):
            raise ValueError("The features listed in data matrix of condition A is not equal to" + 
                             " the features listed in data matrix of condition B!\n" + 
                             Utils.generate_additional_description("first and second argument"))
               
        for i in range(len(condA_mixed_matrix_pd.index)):
            if condA_mixed_matrix_pd.index[i] != condB_mixed_matrix_pd.index[i]:
                raise ValueError("The features listed in data matrix of condition A is not equal to" + 
                                " the features listed in data matrix of condition B!\n" + 
                                Utils.generate_additional_description("first and second argument"))
               
               
        SelectedFeaturesUtils.check(selected_features, condA_mixed_matrix_pd, "third argument")
        ChangeTendencyUtils.check(change_tendency, "fourth argument")               
        
        average_condA_pd = np.mean(condA_mixed_matrix_pd, axis = 1)
        average_condB_pd = np.mean(condB_mixed_matrix_pd, axis = 1)
        
        other_feature = list(set(average_condA_pd.index) - set(selected_features))
        
        log_fold_change_pd = np.log2((1 + average_condB_pd)/(1 + average_condA_pd))
        ranksums_result = stats.ranksums(
            log_fold_change_pd.loc[selected_features], \
            log_fold_change_pd.loc[other_feature])
            
        statistic_result = ranksums_result[0]
        p_value = ranksums_result[1]/2
        
        if change_tendency == "up":
            if statistic_result < 0:
                p_value = 1 - p_value
            
        elif change_tendency == "down":
            if statistic_result > 0:
                p_value = 1 - p_value
        
        return p_value, log_fold_change_pd, average_condA_pd, average_condB_pd
    
    
    @staticmethod
    def validate_data_matrix_specific_noise_ratio(data_matrix_pd,
                                                  noise_matrix,
                                                  cond_pair,
                                                  conditions,
                                                  selected_features,
                                                  change_tendency,
                                                  noise_ratio):
        """
        Run knowledge capture validation with specified noise ratio. This
        method perform the validation with the following steps:
        
        1. Preparing the perturbed matrix by mixing the original
           data matrix with noise matrix with specified noise ratio.
        2. Extracting the data matrix of two conditions with user specified
           condition table (mapping samples to conditions) and condition pair.
        3. Evaluating the fold change of average values between these two 
           conditions.
        4. Performing one-tailed Wilcoxon ranksum test to compare the fold 
           change of selected features and other features.
           
        Finally the p-value will be the output. 
        
        A final score can be evaluated from a series of p-value evaluated from 
        different noise ratio by picking the highest noise ratio that the 
        p-value is still lower than the specified threshold.

        Parameters
        ----------
        data_matrix_pd : A pandas dataframe
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
            
        noise_matrix : A 2D numpy array
            The noise matrix. The shape of the noise matrix should be the same
            as the shape of input compendium.
            
        cond_pair : A tuple with two elements
            The first element will be identified as condition A (control).
            The seocnd element will be identified as condition B 
            (experimental condition).
            
        conditions : A pandas DataFrame
            The condition table with condition labels of samples.
            The table should have two columns. The first column listed the 
            samples (which should be available in input data matrix), and the 
            second column listed the condition labels of samples.
            
            The second column should have the header 'Condition'. In addition,
            the condition table should have at least two conditions and at 
            least there are two samples should be from the same condition.
            
            The condition should contain both two conditions listed in 
            cond_pair.
            
        selected_features : A list of string
            Features which is expected to be significantly increase/decrease
            in condition B compared with condition A.
            
            All features should be existed in the dataframe of condition A and 
            B. In addition, no duplicated features are allowed and it is not
            allowed to select all features listed in the dataframe of condition
            A (and B).
            
        change_tendency : A string. Only "up" and "down" is allowed
            The change tendency will be used in one-tailed Wilcoxon ranksum 
            test for comparing the fold change of selected features and other 
            features.
            
        noise_ratio : A float
            A number between 0.0 to 1.0, 0.0 means no noise, 
            0.5 means linear combination of 50% original data matrix and 50% 
            noise, and 1 means 100% noise.

        Returns
        -------
        p_value : A float
            The p-value of one-tailed Wilcoxon ranksum test of comparison
            between the fold change of selected features and the fold change
            of other features.
            
        log_fold_change_selected: A pandas series
            Log2 fold change of selected features between average values 
            of two conditions
            
        log_fold_change_others: A pandas series
            Log2 fold change of other features between average values 
            of two conditions

        """
        
        MatrixUtils.check_input_pd_dataframe(
            data_matrix_pd, "input compendium", "first argument")
        
        MatrixUtils.check_input_matrix(
            noise_matrix, "noise matrix", "second argument")
        
        MatrixUtils.check_two_matrix_shape_consistent(
            data_matrix_pd, noise_matrix,
            "input data matrix", "noise matrix", 
            "first and second argument")
        
        CondPairUtils.check_cond_pair(cond_pair, "third argument")
        
        InputConditionUtils.check(conditions, cond_pair, data_matrix_pd, "fourth argument")
        
        SelectedFeaturesUtils.check(selected_features, data_matrix_pd, "fifth argument")
        
        ChangeTendencyUtils.check(change_tendency, "6th argument")
        
        NoiseRatioUtils.check_noise_ratio(noise_ratio, "7th argument")
        
        mixed_matrix_pd = data_matrix_pd * (1 - noise_ratio) + noise_matrix * noise_ratio

        sample_cond_A = conditions.loc[conditions.loc[:,'Condition'] == cond_pair[0]].index
        sample_cond_B = conditions.loc[conditions.loc[:,'Condition'] == cond_pair[1]].index
        
        condA_mixed_matrix_pd = mixed_matrix_pd.loc[:,sample_cond_A]
        condB_mixed_matrix_pd = mixed_matrix_pd.loc[:,sample_cond_B]
        
        p_value, log_fold_change_pd, average_condA_pd, average_condB_pd = \
            KnowledgeCaptureValidation.knowledge_capture_evaluation_one_pair(
            condA_mixed_matrix_pd,
            condB_mixed_matrix_pd,
            selected_features,
            change_tendency
            )
            
        other_feature = list(set(average_condA_pd.index) - set(selected_features))
        
        avg_log_fold_change_selected = np.mean(log_fold_change_pd.loc[selected_features])
        std_log_fold_change_selected = np.std(log_fold_change_pd.loc[selected_features])
        
        avg_log_fold_change_others = np.mean(log_fold_change_pd.loc[other_feature])
        std_log_fold_change_others = np.std(log_fold_change_pd.loc[other_feature])
        
        n_selected = len(selected_features)
        n_other = len(other_feature)
        
        return p_value, \
                log_fold_change_pd.loc[selected_features], \
                log_fold_change_pd.loc[other_feature]

    @staticmethod
    def validate_data_matrix(data_matrix_pd,
                             cond_pair,
                            conditions,
                            selected_features,
                            change_tendency,
                            noise_ratio_step = 0.1,
                            n_trial = 100):
        """
        Run knowledge capture validation with a series of different noise 
        ratios generated by user-specified noise_ratio_step. This
        method perform the validation with the following steps:
        
        1. Preparing the perturbed matrix by mixing the original
           data matrix with noise matrix with specified noise ratio.
        2. Extracting the data matrix of two conditions with user specified
           condition table (mapping samples to conditions) and condition pair.
        3. Evaluating the fold change of average values between these two 
           conditions.
        4. Performing one-tailed Wilcoxon ranksum test to compare the fold 
           change of selected features and other features.
           
        Finally the p-value will be the output. 
        
        A final score can be evaluated from a series of p-value evaluated from 
        different noise ratio by picking the highest noise ratio that the 
        p-value is still lower than the specified threshold.
        
        This method perform validation with different noise ratio for multiple
        times. The noise matrix will be generated for each trial by randomly
        permutating the orignal input data matrix.

        Parameters
        ----------
        data_matrix_pd : A pandas dataframe
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
            
        cond_pair : A tuple with two elements
            The first element will be identified as condition A (control).
            The seocnd element will be identified as condition B 
            (experimental condition).
            
        conditions : A pandas DataFrame
            The condition table with condition labels of samples.
            The table should have two columns. The first column listed the 
            samples (which should be available in input data matrix), and the 
            second column listed the condition labels of samples.
            
            The second column should have the header 'Condition'. In addition,
            the condition table should have at least two conditions and at 
            least there are two samples should be from the same condition.
            
            The condition should contain both two conditions listed in 
            cond_pair.
            
        selected_features : A list of string
            Features which is expected to be significantly increase/decrease
            in condition B compared with condition A.
            
            All features should be existed in the dataframe of condition A and 
            B. In addition, no duplicated features are allowed and it is not
            allowed to select all features listed in the dataframe of condition
            A (and B).
            
        change_tendency : A string. Only "up" and "down" is allowed
            The change tendency will be used in one-tailed Wilcoxon ranksum 
            test for comparing the fold change of selected features and other 
            features.
            
        noise_ratio_step : A float, optional
            The noise ratio step for generating the series of noise ratio. 
            The default is 0.1.
            
        n_trial : An integer, optional
            An integer specify how many iterations to be run. One 
            iteration means calling validate_data_matrix_specific_noise_ratio() 
            with noise ratio from 0 to 1 with specified noise ratio step.
            The default is 100.

        Returns
        -------
        result : A pandas dataframe
            The evaluated p-values of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.     
        avg_selected_log_fc : A pandas dataframe
            The average log2 fold change of selected features of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.  
        avg_others_log_fc : A pandas dataframe
            The average log2 fold change of other features of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        std_selected_log_fc : A pandas dataframe
            The standard deviation of log2 fold change of selected features 
            of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials. 
        std_others_log_fc : A pandas dataframe
            The standard deviation of log2 fold change of other features 
            of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        n_selected : An integer
            The number of selected features
        n_others : An integer
            The number of other (non-selected) features

        """
        
        MatrixUtils.check_input_pd_dataframe(
            data_matrix_pd, "input compendium", "first argument")
        
        CondPairUtils.check_cond_pair(cond_pair, "second argument")
        
        InputConditionUtils.check(
            conditions, cond_pair, data_matrix_pd, "third argument")
        
        SelectedFeaturesUtils.check(
            selected_features, data_matrix_pd, "fourth argument")
        
        ChangeTendencyUtils.check(change_tendency, "fifth argument")
        
        NoiseRatioUtils.check_noise_ratio_step(
            noise_ratio_step, "6th argument")
        
        NTrialUtils.check_n_trial(n_trial, "7th argument")
        
        noise_ratio_list = np.arange(0.0,1.1,noise_ratio_step)
        
        result = pd.DataFrame(0, index = noise_ratio_list, columns = 
                              np.arange(n_trial))
        avg_selected_log_fc = pd.DataFrame(0, index = noise_ratio_list, columns = 
                              np.arange(n_trial))
        avg_others_log_fc = pd.DataFrame(0, index = noise_ratio_list, columns = 
                              np.arange(n_trial))
        std_selected_log_fc = pd.DataFrame(0, index = noise_ratio_list, columns = 
                              np.arange(n_trial))
        std_others_log_fc = pd.DataFrame(0, index = noise_ratio_list, columns = 
                              np.arange(n_trial))
        
        for x in range(n_trial):
            noise_matrix = Utils.prepare_noise(data_matrix_pd.values)
            for noise_ratio in noise_ratio_list:
                p_value, log_fold_change_selected, log_fold_change_others = \
                    KnowledgeCaptureValidation.validate_data_matrix_specific_noise_ratio(
                    data_matrix_pd, 
                    noise_matrix, 
                    cond_pair, 
                    conditions, 
                    selected_features, 
                    change_tendency, 
                    noise_ratio)

                result.loc[noise_ratio, x] = p_value
                avg_selected_log_fc.loc[noise_ratio, x] = np.mean(log_fold_change_selected)
                avg_others_log_fc.loc[noise_ratio, x] = np.mean(log_fold_change_others)
                std_selected_log_fc.loc[noise_ratio, x] = np.std(log_fold_change_selected)
                std_others_log_fc.loc[noise_ratio, x] = np.std(log_fold_change_others)
                
        
        return result, avg_selected_log_fc, avg_others_log_fc, \
                std_selected_log_fc, std_others_log_fc, \
                len(log_fold_change_selected), len(log_fold_change_others)
         
        
    def plot_result(result, filename, **kwargs):
        """
        
        Plot the noise ratio to p-values of all trials.

        Parameters
        ----------
        result : A pandas dataframe
            The evaluated p-values of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        filename : A String
            A string that specify the image file path to be saved
            No file will be saved if it is None.
        **kwargs : 
            Additional arguments for matplotlib.pyplot.savefig()

        Returns
        -------
        None.

        """
        
        """Check result
        """
        ResultUtils.check_knowledge_capture_validation_result(result, "first argument")
        
        fig = plt.figure(dpi = 200)
        knowledge_capture_plot = result.plot(
            title='Knowledge capture validation results (' + 
            str(result.shape[1]) + 
            ' trials)',
            legend = False, 
            xlabel='Noise Ratio', ylabel='p-Value')
        
        if filename is not None:
            plt.savefig(str(filename), **kwargs)
        plt.close(fig)
        
    def parse_result(result, pval_threshold=0.05):
        """
        
        Evaluate the score of the knowledge capture validation of the 
        compendium of trials

        Parameters
        ----------
        result : A pandas dataframe
            The evaluated p-values of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        pval_threshold : A float
            The threshold of the p-value. The default is 0.05.

        Returns
        -------
        scores : A pandas dataframe
            The score matrix has one column 'Score' that records the scores 
            of trials.

        """
        
        """Check result
        """
        ResultUtils.check_knowledge_capture_validation_result(result, 
                                                        "first argument")
        
        
        PValueUtils.check_pvalue_threshold(pval_threshold, 
                                           "second argument")
            
        scores = pd.DataFrame(0,index=result.columns,columns=['Score'])
        for col in result.columns:
            scores.loc[col,'Score'] = 0
            for idx in result.index:
                if result.loc[idx,col] >= pval_threshold:
                    break
                scores.loc[col,'Score'] = idx
    
        return scores
    
    def plot_avg_log_fold_changes(result, scores, 
                      avg_selected_log_fc, avg_others_log_fc, 
                      std_selected_log_fc, std_others_log_fc, 
                      n_selected, n_others, idx_trial, filename, **kwargs):
        """
        Plot the p-values and log2 fold changes of selected/non-selected 
        features of different noise ratio.

        Parameters
        ----------
        result : A pandas dataframe
            The evaluated p-values of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        scores : A pandas dataframe
            The score matrix has one column 'Score' that records the scores 
            of trials.
        avg_selected_log_fc : A pandas dataframe
            The average log2 fold change of selected features of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.  
        avg_others_log_fc : A pandas dataframe
            The average log2 fold change of other features of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        std_selected_log_fc : A pandas dataframe
            The standard deviation of log2 fold change of selected features 
            of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials. 
        std_others_log_fc : A pandas dataframe
            The standard deviation of log2 fold change of other features 
            of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        n_selected : An integer
            The number of selected features
        n_others : An integer
            The number of other (non-selected) features
        idx_trial : An integer
            An integer specify the trial index
        filename : A String
            A string that specify the image file path to be saved
            No file will be saved if it is None.
        **kwargs : 
            Additional arguments for matplotlib.pyplot.savefig()

        Returns
        -------
        None.

        """
        
        ResultUtils.check_knowledge_capture_validation_result(result, 
                                                        "first argument")
        ResultUtils.check_general_score(scores, 
                                        "knowledge capture validation score", 
                                        "second argument")
        
        ResultUtils.check_general_result(avg_selected_log_fc, 
                                         "average fold change (selected features)", 
                                         "third argument")
        ResultUtils.check_general_result(avg_others_log_fc, 
                                         "average fold change (other features)", 
                                         "fourth argument")
        ResultUtils.check_general_result(std_selected_log_fc, 
                                         "std. of fold change (selected features)", 
                                         "fifth argument")
        ResultUtils.check_general_result(std_others_log_fc, 
                                         "std. of fold change (other features)", 
                                         "6th argument")
        
        Utils.check_int(n_selected, "#selected features should be an integer!",
                        "7th argument")
        Utils.check_value_boundary(n_selected <= 0, 
                                   "#selected features should be positive!",
                        "7th argument")
        
        Utils.check_int(n_others, "#other features should be an integer!",
                        "8th argument")
        Utils.check_value_boundary(n_others <= 0, 
                                   "#other features should be positive!",
                        "8th argument") 
        

        NTrialUtils.check_trial_index(idx_trial, result, "9th argument")
        
        
        fig = plt.figure(dpi = 200)
        ax = fig.add_axes([0,0,1,1])
        ax2 = ax.twinx()
        ax.plot(result.index, result.loc[:,idx_trial], color=(0,0,1,1))
        ax.set_ylabel('p-value')
        ax.set_xlabel('Noise Ratio')
        ax.yaxis.label.set_color('blue')
        ax.tick_params(axis='y', colors='blue')
        ax.spines['left'].set_color('blue')  
        
        ax2.set_title('pValue and average log2 fold changes\n(Selected features VS Other features)\n' + \
                     '#Selected features = ' + str(n_selected) + \
                         ', #Other features = ' + str(n_others) + \
                         '\nScores = ' + str(scores.loc[idx_trial,'Score']) )
            
        
        ax2.set_ylabel('Log2 average fold change')
        
        ax2.plot(avg_others_log_fc.index,avg_others_log_fc.loc[:,idx_trial],color='black',linewidth=3,
                label='Log2 average fold change (Other features)')
        ax2.plot(avg_others_log_fc.index,avg_others_log_fc.loc[:,idx_trial] + std_others_log_fc.loc[:,idx_trial],color='black',linewidth=1)
        ax2.plot(avg_others_log_fc.index,avg_others_log_fc.loc[:,idx_trial] - std_others_log_fc.loc[:,idx_trial],color='black',linewidth=1)
        ax2.fill_between(avg_others_log_fc.index,
                        avg_others_log_fc.loc[:,idx_trial] - std_others_log_fc.loc[:,idx_trial],
                        avg_others_log_fc.loc[:,idx_trial] + std_others_log_fc.loc[:,idx_trial],
                        color=(0,0,0,0.3))
        
        ax2.plot(avg_selected_log_fc.index,avg_selected_log_fc.loc[:,idx_trial],color='red',linewidth=3,
                label='Log2 average fold change (Selected features)')
        ax2.plot(avg_selected_log_fc.index,avg_selected_log_fc.loc[:,idx_trial] + std_selected_log_fc.loc[:,idx_trial],color='red',linewidth=1)
        ax2.plot(avg_selected_log_fc.index,avg_selected_log_fc.loc[:,idx_trial] - std_selected_log_fc.loc[:,idx_trial],color='red',linewidth=1)
        ax2.fill_between(avg_selected_log_fc.index, 
                        avg_selected_log_fc.loc[:,idx_trial] - std_selected_log_fc.loc[:,idx_trial],
                        avg_selected_log_fc.loc[:,idx_trial] + std_selected_log_fc.loc[:,idx_trial],
                        color=(1,0,0,0.3))
        
        
        ax2.legend(loc='upper right', framealpha = 1.0) 
        if filename is not None:
            plt.savefig(str(filename), bbox_inches='tight', pad_inches=0.1, **kwargs)
        plt.close(fig)
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     "Evaluate the quality of an omics\n" + 
                                     "compendium using knowledge cpature validation approach")
    
    parser.add_argument('input_compendium_path', type=str,
                        help='The input path of the omics compendium.\n' + 
                             'Rows represent features and Columns represent samples.')
    
    parser.add_argument('input_condition_label_path', type=str,
                        help='The condition table contains condition label for each sample.' +
                             'The table should have two columns: The first column is the sample' +
                             ' label, and the second column is the corresponded condition labels.\n' +
                             "The second column must have a header 'Condition'.")
    
    parser.add_argument('control_condition', type=str,
                        help='The condition label (control). The log2 fold change will be evaluated by' + 
                             ' the formula : log2((1 + feature_control) / (1 + feature_experiment))')
    
    parser.add_argument('experimental_condition', type=str,
                        help='The condition label (experimental case). The log2 fold change will be evaluated by' + 
                             ' the formula : log2((1 + feature_control) / (1 + feature_experiment))')
    
    parser.add_argument('input_selected_features_path', type=str,
                        help='The features which are expected to be significantly increase/decrease' + 
                             ' for the experimental case compared with the control case.\n' + 
                             'The first column of the input will be used as input feature.')
    
    parser.add_argument('change_tendency', choices = ['up','down'],
                        help = 'The expected change direction of experiment condition compared with control.\n' + 
                               "The option is 'up' or 'down'.")
    
    parser.add_argument('--n_trial', type=int, default = 100,
                        help='The number of trial.')
    
    parser.add_argument('--noise_ratio_step', type=float, default = 0.01,
                        help='Noise ratio step. The validation procedure will evaluate' + 
                        'the imputation error of different ratio from 0 to 1 with' + 
                        'this specified noise ratio step.')
    
    parser.add_argument('--p_value_threshold', type=float, default = 0.05,
                        help='Threshold of p-value to indicate the significance' +
                             ' of the difference between the average fold change of selected features' + 
                             ' and the average fold change of other features.')
    
    parser.add_argument('--output_pvalue_table_path', type=str, default = None,
                        help='The path of the output p-value table of N trials (in csv format).\n' + 
                        'No file will be produced if it is not provided.')
    
    parser.add_argument('--output_score_path', type=str, default = None,
                        help='The path of the score table of N trials (in csv format).\n' + 
                        'The score is the highest noise ratio that the corresponded mixed matrix' +
                        ' can still yield significance difference of avg fold change of selected features' +
                        ' and the avg fold change of other features.\n' + 
                        'The p-value threshold can be specified by users (default is 0.05)\n' +
                        'No file will be produced if it is not provided.')                 
    
    parser.add_argument('--plot_overall_result_path', type=str, default = None,
                        help='The path of the plot of noise ratio to p-value of N trials.\n' + 
                        'No figure will be plotted if it is not provided.')
    parser.add_argument('--plot_saving_format', type=str, default = 'svg',
                        help='The file format of the saving plot.\n' + 
                        'Default is svg file.')

    parser.add_argument('--plot_single_trial_result_path', type=str, default = None,
                        help='The path of the plot of a single trial result.\n' + 
                        'It plot the avg fold change of selected features and' + 
                        ' avg fold change of other features for different noise ratios.\n' + 
                        'No figure will be plotted if it is not provided.')
    
    parser.add_argument('--trial_idx', type=int, default = 0,
                        help='The trial index for plotting.\n' + 
                        'Default is 0 (i.e. the first trial result will be plotted).')
    parser.add_argument('--demo', action = 'store_true',
                        help='To run the demo or not (true or false). If true,' +
                        ' only the first 20 samples listed in the condition table will be used and'
                        ' #trials will be set as 1 and noise ratios step will be set as 0.2' + 
                        ' to reduce the computation time for demo')                    
    
    args = parser.parse_args() 
    
    input_csv = args.input_compendium_path
    data_matrix_pd = pd.read_csv(input_csv, index_col = 0)
    
    conditions = pd.read_csv(args.input_condition_label_path, index_col = 0)
    selected_features_pd = pd.read_csv(args.input_selected_features_path, index_col = 0)
    selected_features = selected_features_pd.index.tolist()
    
    cond_A = args.control_condition
    cond_B = args.experimental_condition
    
    change_tendency = args.change_tendency
    
    if args.demo is True: 
        print("Running Demo Mode!")
        args.noise_ratio_step = 0.1
        args.n_trial = 3
    
    result, avg_selected_log_fc, avg_others_log_fc, std_selected_log_fc, std_others_log_fc, \
    n_selected, n_others = KnowledgeCaptureValidation.validate_data_matrix(
        data_matrix_pd,
        (cond_A, cond_B), 
        conditions, 
        selected_features, 
        change_tendency,
        args.noise_ratio_step,
        args.n_trial)
    
    scores = KnowledgeCaptureValidation.parse_result(result, args.p_value_threshold)
    
    print("Knowledge capture validation result of input " + input_csv + " : ")
    print("Compendium size (#features, #samples) : " + str(data_matrix_pd.shape))
    n_control_samples = len(conditions.loc[conditions.loc[:,'Condition'] == cond_A,:].index)
    n_exp_samples = len(conditions.loc[conditions.loc[:,'Condition'] == cond_B,:].index)
    
    print("#Control Condition Samples : " + str(n_control_samples))
    print("#Experimental Condition Samples : " + str(n_exp_samples))
    
    print("#selected features : " + str(n_selected))
    print("#non-selected features : " + str(n_others))
    
    print("Average score of " + str(args.n_trial) + " trials : " + 
          "{:.3f}".format(np.mean(scores.loc[:,'Score'])) + " +/- " + 
          "{:.3f}".format(np.std(scores.loc[:,'Score'])))
    
    ''' For testing output files
    args.output_pvalue_table_path = "test_knowledge_capature_pvalue.csv"
    args.output_score_path = "test_knowledge_capture_score.csv"
    args.plot_overall_result_path = "test_knowledge_capture_plotall.svg"
    args.plot_single_trial_result_path = "test_knowledge_capture_plot_single.svg"
    '''
    
    if args.output_pvalue_table_path is not None:
        result.to_csv(args.output_pvalue_table_path)
        
    if args.output_score_path is not None:
        scores.to_csv(args.output_score_path)    
    
    if args.plot_overall_result_path is not None:
        KnowledgeCaptureValidation.plot_result(result, 
                                          args.plot_overall_result_path, 
                                          format=args.plot_saving_format)
        
    if args.plot_single_trial_result_path is not None:
        KnowledgeCaptureValidation.plot_avg_log_fold_changes( 
                                            result, scores,
                                            avg_selected_log_fc,avg_others_log_fc,
                                            std_selected_log_fc,std_others_log_fc,\
                      n_selected, n_others, args.trial_idx, 
                      args.plot_single_trial_result_path, 
                      format=args.plot_saving_format)
    
    