# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:46:33 2021

CorrelationValidation.py: 
Evaluate the quality of an omics compendium with condition labels.

This approach assumes that the features (i.e. gene expressions) of samples
from the same condition should be more similar than the features from different
samples. Therefore, the pairwise Pearson Correlation Coefficients (PCC) among 
samples from the same condition are expected to be higher than pairwise PCC
among samples from all conditions. The assumption can be verified by performing
one-tailed Wilcoxon Ranksum test between PCCs among same condition samples and
PCCs among all samples.

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

class InputConditionUtils():
    '''
    Utilities for checking the input condition table
    
    Methods in this class will be called internally for checking the input
    '''
    
    @staticmethod
    def check(
        conditions,
        input_matrix,
        argument_order = "first argument"):
            
            InputConditionUtils.check_type(
                conditions, argument_order)
            InputConditionUtils.check_invalid_shape(
                conditions, argument_order)
            InputConditionUtils.check_empty(
                conditions, argument_order)
            InputConditionUtils.check_only_one_condition(
                conditions, argument_order)
            InputConditionUtils.check_all_samples_from_different_conditions(
                conditions, input_matrix, argument_order)
            InputConditionUtils.check_invalid_element_num(
                conditions, input_matrix, argument_order)
            InputConditionUtils.check_input_condition_invalid_element_type_message(
                conditions, argument_order)

    @staticmethod
    def check_type(
        conditions, 
        argument_order = "first argument"):
        
        if type(conditions) is not np.ndarray:
            raise TypeError(
                  "Input condition for correlation validation" + 
                  " is not a numpy n-dimension array!\n" + 
                  Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def check_invalid_shape(
        conditions, 
        argument_order = "first argument"):
        
        if len(conditions.shape) != 1:
            raise ValueError(
                  "The dimension of input condition" + 
                  " for correlation validation" + 
                  " should be 1D!\n" + 
                  Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def check_empty(
        conditions, 
        argument_order = "first argument"):
        
        if len(conditions) == 0:
            raise ValueError(
                  "The input condition for correlation validation is empty!\n" + 
                  Utils.generate_additional_description(argument_order))
                  
    @staticmethod
    def check_only_one_condition(
        conditions,
        argument_order = "first argument"):
        
        if len(set(conditions)) == 1:
            raise ValueError(
                  "All samples are from the same conditions!\n" +
                  "At least two conditions should be provided.\n" + 
                  Utils.generate_additional_description(argument_order))
            
    @staticmethod
    def check_all_samples_from_different_conditions(
        conditions,
        input_matrix,
        argument_order = "first argument"):
        
        if len(set(conditions)) == input_matrix.shape[1]:
            raise ValueError(
                  "All samples are from different conditions!\n" +
                  "At least two samples should be from" + 
                  " the same conditions.\n" +  
                  Utils.generate_additional_description(argument_order))
        
    @staticmethod
    def check_invalid_element_num(
        conditions,
        input_matrix,
        argument_order = "first argument"):
         
        if len(conditions) != input_matrix.shape[1]:
           raise ValueError(
                 "The #element in the input condition" + 
                 " for correlation validation" + 
                 " is not equal to the #samples in the data matrix.\n" +
                 Utils.generate_additional_description(argument_order))
                 
    @staticmethod
    def check_input_condition_invalid_element_type_message(
        conditions, 
        argument_order = "first argument"):
        
        for element in conditions:
            if type(element) is not np.str_ and type(element) is not str:
                raise TypeError(
                      "At least one element in the input condition" + 
                      " for correlation validation is not a String!\n" +
                      Utils.generate_additional_description(argument_order))
                      



class CorrelationValidation:
    @staticmethod
    def correlation_evaluation(data_matrix, conditions):
        """
        
        Evaluate the pairwise Pearson Correlation Coefficients (PCC) of the 
        data matrix among all samples, and group the PCC of pairs from the same 
        conditions and different conditions, finally evaluate the significant
        of difference of average pairwise PCC among the same condition samples 
        and average pairwise PCC of all samples.
        
        It is expected that the average PCC among the same condition should be
        higher than the average PCC of all samples

        Parameters
        ----------
        data_matrix : A 2D numpy array
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
            
        conditions : A pandas DataFrame
            The condition table with condition labels of samples.
            The table should have two columns. The first column listed the 
            samples (which should be available in input data matrix), and the 
            second column listed the condition labels of samples.
            
            The second column should have the header 'Condition'. In addition,
            the condition table should have at least two conditions and at 
            least there are two samples should be from the same condition.

        Returns
        -------
        pvalue : A float (np.float)
            The p-value which indicate the significance of the difference of 
            average pairwise PCC from the same condition samples and average 
            pairwise PCC from all samples. One-tailed Wilcoxon ranksum test
            will be performed.
        corr_all : A list of floats (np.float)
            The list contains the pairwise PCCs from all samples. Note that
            len(corr_all) = len(corr_same_condition) + len(corr_diff_condition)
        corr_same_condition : A list of floats (np.float)
            The list contains the pairwise PCCs from the same condition samples
        corr_diff_condition : A list of floats (np.float)
            The list contains the pairwise PCCs from the different condition 
            samples

        """
        
        MatrixUtils.check_input_matrix(data_matrix)
        InputConditionUtils.check(
            conditions, data_matrix, "second argument")
        
        corr_all = []
        corr_diff_condition = []
        corr_same_condition = []
        
        for i in range(data_matrix.shape[1]):
            x = np.array(data_matrix[:,i])
            for j in range(data_matrix.shape[1]):
                if (i <= j):
                    continue
                
                y = np.array(data_matrix[:,j])
                corr = np.corrcoef(x,y)[0,1]
                corr_all.append(corr)
                if (conditions[i] == conditions[j]):
                    corr_same_condition.append(corr)
                else:
                    corr_diff_condition.append(corr)
        
        statistic, pvalue = stats.ttest_ind(corr_all, corr_same_condition, equal_var = False)
        if (np.mean(corr_all) < np.mean(corr_same_condition)):
            pvalue = pvalue * 0.5
        else:
            pvalue = 1 - pvalue * 0.5
                 
        return pvalue, corr_all, corr_same_condition, corr_diff_condition
    
    @staticmethod
    def validate_data_matrix_specific_noise_ratio(data_matrix,
                                                  conditions,
                                                  noise_matrix,
                                                  noise_ratio):
        """
        
        Evaluate the pairwise PCCs among samples with specified noise ratio.
        This method calls correlation_evaluation() with a perturb matrix which
        is generated by the linear combination of original data matrix and 
        noise matrix (which can be generated by randomly permuting the elements
        of the original data matrix using Utils.prepare_noise())
        

        Parameters
        ----------
        data_matrix : A 2D numpy array
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
        conditions : A pandas DataFrame
            The condition table with condition labels of samples.
            The table should have two columns. The first column listed the 
            samples (which should be available in input data matrix), and the 
            second column listed the condition labels of samples.
            
            The second column should have the header 'Condition'. In addition,
            the condition table should have at least two conditions and at 
            least there are two samples should be from the same condition.
            DESCRIPTION.
        noise_matrix : A 2D numpy array
            A noise matrix which can be generated by Utils.prepare_noise() in 
            this module by randomly permute the input data matrix.
            Matrices with NaN value and infinite values are not allowed.
        noise_ratio : A float
            A number between 0.0 to 1.0, 0.0 means no noise, 
            0.5 means linear combination of 50% original data matrix and 50% 
            noise, and 1 means 100% noise.

        Returns
        -------
        pvalue : A float (np.float)
            The p-value which indicate the significance of the difference of 
            average pairwise PCC from the same condition samples and average 
            pairwise PCC from all samples. One-tailed Wilcoxon ranksum test
            will be performed.
        corr_all : A list of floats (np.float)
            The list contains the pairwise PCCs from all samples. Note that
            len(corr_all) = len(corr_same_condition) + len(corr_diff_condition)
        corr_same_condition : A list of floats (np.float)
            The list contains the pairwise PCCs from the same condition samples
        corr_diff_condition : A list of floats (np.float)
            The list contains the pairwise PCCs from the different condition 
            samples

        """
                                                  
        MatrixUtils.check_input_matrix(data_matrix)
        InputConditionUtils.check(
            conditions, data_matrix, "second argument")
        MatrixUtils.check_input_matrix(
            noise_matrix, "noise matrix", "third argument")
        NoiseRatioUtils.check_noise_ratio(noise_ratio, "fourth argument")
        
        MatrixUtils.check_two_matrix_shape_consistent(
            data_matrix, noise_matrix, "input matrix", "noise matrix", 
            "first and third argument")
        
        mixed_matrix = noise_matrix * noise_ratio + \
                       data_matrix * (1 - noise_ratio)
  
        pvalue, corr_all, corr_same_condition, corr_diff_condition = \
            CorrelationValidation.correlation_evaluation(mixed_matrix, conditions)    
        return pvalue, corr_all, corr_same_condition, corr_diff_condition
    
    @staticmethod
    def validate_data_matrix(data_matrix,
                            conditions,
                            noise_ratio_step = 0.1,
                            n_trial = 100):
        """
        
        Run the correlation validation of the compendium.
        
        Evaluate p-values which indicate the significance of difference between
        average pairwise PCC among same condition samples and average pairwise 
        PCC among all samples with a sequence of noise ratios for multiple times
        
        This is the main method of correlation validation module. Once the 
        p-values are evaluated, the scores can be evaluated by finding the
        highest noise ratio which the corresponded perturbed matrix can yields
        significance results.
        
        
        Parameters
        ----------
        data_matrix : A 2D numpy array
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
        conditions : A pandas DataFrame
            The condition table with condition labels of samples.
            The table should have two columns. The first column listed the 
            samples (which should be available in input data matrix), and the 
            second column listed the condition labels of samples.
            
            The second column should have the header 'Condition'. In addition,
            the condition table should have at least two conditions and at 
            least there are two samples should be from the same condition.
            DESCRIPTION.
        noise_ratio_step : A float, optional
            A number between 0 to 0.5 (0 is not allowed)
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
        avg_corr : A pandas dataframe
            The average pairwise PCC of all samples of different trials and
            noise ratios.
            Each row represent different ratios and each column represent 
            different trials.  
        avg_same_corr : A pandas dataframe
            The average pairwise PCC of same condition samples of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        std_corr : A pandas dataframe
            The standard deviation of pairwise PCC of all samples of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials. 
        std_same_corr : A pandas dataframe
            The standard deviation of pairwise PCC of same condition samples 
            of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        n_corr : An integer
            The number of pairwise PCC of all samples
        n_same_corr : An integer
            The number of pairwise PCC of same condition samples
        """
        
        MatrixUtils.check_input_matrix(data_matrix)
        InputConditionUtils.check(
            conditions, data_matrix, "second argument")
        NoiseRatioUtils.check_noise_ratio_step(noise_ratio_step, "third argument")
        NTrialUtils.check_n_trial(n_trial, "fourth argument")
     
        noise_ratios = np.arange(0.0, 1.01, noise_ratio_step)
        result = pd.DataFrame(0, index = noise_ratios, columns = 
                              np.arange(n_trial))
        avg_corr = pd.DataFrame(0, index = noise_ratios, columns = 
                              np.arange(n_trial))
        avg_same_corr = pd.DataFrame(0, index = noise_ratios, columns = 
                              np.arange(n_trial))
        std_corr = pd.DataFrame(0, index = noise_ratios, columns = 
                              np.arange(n_trial))
        std_same_corr = pd.DataFrame(0, index = noise_ratios, columns = 
                              np.arange(n_trial))
        
        for x in np.arange(n_trial):
            noise_matrix = Utils.prepare_noise(data_matrix)
            for noise_ratio in noise_ratios: 
                pvalue, corr_all, corr_same_condition, corr_diff_condition = \
                    CorrelationValidation.validate_data_matrix_specific_noise_ratio(
                        data_matrix, 
                        conditions,
                        noise_matrix,
                        noise_ratio)   
                 
                result.loc[noise_ratio,x] = pvalue
                avg_corr.loc[noise_ratio,x] = np.mean(corr_all)
                avg_same_corr.loc[noise_ratio,x] = np.mean(corr_same_condition)
                std_corr.loc[noise_ratio,x] = np.std(corr_all)
                std_same_corr.loc[noise_ratio,x] = np.std(corr_same_condition)
        
        return result, avg_corr, avg_same_corr, std_corr, std_same_corr, \
                len(corr_all), len(corr_same_condition)
    
    
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
            A string that specify the image file path to be saved.
            No file will be saved if it is None.
        **kwargs : 
            Additional arguments for matplotlib.pyplot.savefig()

        Returns
        -------
        None.

        """
        
        """Check result
        """
        ResultUtils.check_correlation_validation_result(result, "first argument")
        
        fig = plt.figure(dpi = 200)
        correlation_plot = result.plot(
            title='Correlation validation results (' + 
            str(result.shape[1]) + 
            ' trials)',
            legend = False, 
            xlabel='Noise Ratio', ylabel='p-Value')
        
        if filename is not None:
            plt.savefig(str(filename), **kwargs)
        plt.close(fig)
        
    def parse_result(result, pval_threshold=0.05):
        """
        
        Evaluate the score of the correlation validation of the compendium of
        trials

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
        ResultUtils.check_correlation_validation_result(result, 
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
    
    def plot_avg_corr(result, scores, avg_corr, avg_same_corr, std_corr, std_same_corr, 
                      n_corr, n_same_corr, idx_trial, filename, **kwargs):
        """
        
        Plot the p-values and average pairwise PCCs among same condition 
        samples and average airwise PCCs among all samples of different noise
        ratio.

        Parameters
        ----------
        result : A pandas dataframe
            The evaluated p-values of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        scores : A pandas dataframe
            The score matrix has one column 'Score' that records the scores 
            of trials.
        avg_corr : A pandas dataframe
            The average pairwise PCC of all samples of different trials and
            noise ratios.
            Each row represent different ratios and each column represent 
            different trials.  
        avg_same_corr : A pandas dataframe
            The average pairwise PCC of same condition samples of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        std_corr : A pandas dataframe
            The standard deviation of pairwise PCC of all samples of different 
            trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials. 
        std_same_corr : A pandas dataframe
            The standard deviation of pairwise PCC of same condition samples 
            of different trials and noise ratios.
            Each row represent different ratios and each column represent 
            different trials.
        n_corr : An integer
            The number of pairwise PCC of all samples
        n_same_corr : An integer
            The number of pairwise PCC of same condition samples
        idx_trial : An integer
            An integer specify the trial index
        filename : A String
            A string that specify the image file path to be saved
        **kwargs : 
            Additional arguments for matplotlib.pyplot.savefig()

        Returns
        -------
        None.

        """
        
        
        ResultUtils.check_correlation_validation_result(result, 
                                                        "first argument")
        ResultUtils.check_general_score(scores, 
                                        "correlation validation score", 
                                        "second argument")
        
        ResultUtils.check_general_result(avg_corr, 
                                         "average correlation matrix (all condition)", 
                                         "third argument")
        ResultUtils.check_general_result(avg_same_corr, 
                                         "average correlation matrix (same condition)", 
                                         "fourth argument")
        ResultUtils.check_general_result(std_corr, 
                                         "std. of correlation matrix (all conditions)", 
                                         "fifth argument")
        ResultUtils.check_general_result(std_same_corr, 
                                         "std. of correlation matrix (same conditions)", 
                                         "6th argument")
        
        Utils.check_int(n_corr, "#correpation pairs of the entire dataset should be an integer!",
                        "7th argument")
        Utils.check_value_boundary(n_corr <= 0, 
                                   "#correpation pairs of the entire dataset should be positive!",
                        "7th argument")
        
        Utils.check_int(n_same_corr, "#correpation pairs from the same conditions should be an integer!",
                        "8th argument")
        Utils.check_value_boundary(n_same_corr <= 0, 
                                   "#correpation pairs from the same conditions should be positive!",
                        "8th argument") 
        
        Utils.check_value_boundary(n_same_corr > n_corr, 
                                   "#correpation pairs of the entire dataset should be greater than " + 
                                   "#correlation pairs from the same conditions",
                        "7th and 8th argument") 
        
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
        
        ax2.set_title('pValue and Pairwise PCC (All samples VS Same Condition samples)\n' + \
                     '#All Sample Pairs = ' + str(n_corr) + \
                         ', #Same Sample Condition Pairs = ' + str(n_same_corr) + \
                         '\nScores = ' + str(scores.loc[idx_trial,'Score']) )
            
        
        ax2.set_ylabel('PCC')
        
        ax2.plot(avg_corr.index,avg_corr.loc[:,idx_trial],color='black',linewidth=3,
                label='Average PCC (All Sample Pairs)')
        ax2.plot(avg_corr.index,avg_corr.loc[:,idx_trial] + std_corr.loc[:,idx_trial],color='black',linewidth=1)
        ax2.plot(avg_corr.index,avg_corr.loc[:,idx_trial] - std_corr.loc[:,idx_trial],color='black',linewidth=1)
        ax2.fill_between(avg_corr.index,
                        avg_corr.loc[:,idx_trial] - std_corr.loc[:,idx_trial],
                        avg_corr.loc[:,idx_trial] + std_corr.loc[:,idx_trial],
                        color=(0,0,0,0.3))
        
        ax2.plot(avg_same_corr.index,avg_same_corr.loc[:,idx_trial],color='red',linewidth=3,
                label='Average PCC (Same Condition Sample Pairs)')
        ax2.plot(avg_same_corr.index,avg_same_corr.loc[:,idx_trial] + std_same_corr.loc[:,idx_trial],color='red',linewidth=1)
        ax2.plot(avg_same_corr.index,avg_same_corr.loc[:,idx_trial] - std_same_corr.loc[:,idx_trial],color='red',linewidth=1)
        ax2.fill_between(avg_same_corr.index, 
                        avg_same_corr.loc[:,idx_trial] - std_same_corr.loc[:,idx_trial],
                        avg_same_corr.loc[:,idx_trial] + std_same_corr.loc[:,idx_trial],
                        color=(1,0,0,0.3))
        
        
        ax2.legend(loc='upper right', framealpha = 1.0) 
        
        if filename is not None:
            plt.savefig(str(filename), bbox_inches='tight', pad_inches=0.1, **kwargs)
        plt.close(fig)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     "Evaluate the quality of an omics\n" + 
                                     "compendium using correlation validation approach")
    
    parser.add_argument('input_compendium_path', type=str,
                        help='The input path of the omics compendium.\n' + 
                             'Rows represent features and Columns represent samples.')
    
    parser.add_argument('input_condition_label_path', type=str,
                        help='The condition table contains condition label for each sample.' +
                             'The table should have two columns: The first column is the sample' +
                             ' label, and the second column is the corresponded condition labels.\n' +
                             "The second column must have a header 'Condition'.")
    
    parser.add_argument('--n_trial', type=int, default = 100,
                        help='The number of trial.')
    
    parser.add_argument('--noise_ratio_step', type=float, default = 0.01,
                        help='Noise ratio step. The validation procedure will evaluate' + 
                        'the imputation error of different ratio from 0 to 1 with' + 
                        'this specified noise ratio step.')
    
    parser.add_argument('--p_value_threshold', type=float, default = 0.05,
                        help='Threshold of p-value to indicate the significance' +
                             ' of the difference between the average PCC among' + 
                             ' the same condition samples and the average PCC among all' +
                             ' samples.')
    
    parser.add_argument('--output_pvalue_table_path', type=str, default = None,
                        help='The path of the output p-value table of N trials (in csv format).\n' + 
                        'No file will be produced if it is not provided.')
    
    parser.add_argument('--output_score_path', type=str, default = None,
                        help='The path of the score table of N trials (in csv format).\n' + 
                        'The score is the highest noise ratio that the corresponded mixed matrix' +
                        ' can still yield significance difference of avg PCC among the same' +
                        ' condition samples and the avg PCC among all samples.\n' + 
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
                        'It plot the avg PCC among the same samples and' + 
                        ' avg PCC among all samples for different noise ratios.\n' + 
                        'No figure will be plotted if it is not provided.')
    
    parser.add_argument('--trial_idx', type=int, default = 0,
                        help='The trial index for plotting.\n' + 
                        'Default is 0 (i.e. the first trial result will be plotted).')
    parser.add_argument('--demo', action='store_true',
                        help='To run the demo. If set,' +
                        ' #trials will be set as 3 and noise ratios step will be set as 0.2' + 
                        ' to reduce the computation time for demo')                    
    
    args = parser.parse_args() 
    
    input_csv = args.input_compendium_path
    data_matrix_pd = pd.read_csv(input_csv, index_col = 0)
    
    input_conditions = args.input_condition_label_path
    condition_matrix = pd.read_csv(input_conditions, index_col = 0)
    
    if 'Condition' not in condition_matrix.columns:
        raise ValueError("The condition table should have " + 
                         "at least one column with label 'Condition'" + 
                         " which labeled the condition of each sample")
    
    for x in condition_matrix.index:
        if x not in data_matrix_pd.columns:
            raise ValueError("One or more samples" + 
                             " listed in the condition table are missing in input compendium\n")
            
    if args.demo is True:
        print("Running Demo Mode!")
        args.n_trial = 3
        args.noise_ratio_step = 0.2    
    
    data_matrix_pd = data_matrix_pd.loc[:,condition_matrix.index]
    data_matrix = np.array(data_matrix_pd)
    conditions = np.array(condition_matrix.loc[data_matrix_pd.columns, 'Condition'])
    
    result, avg_corr, avg_same_corr, std_corr, std_same_corr,\
        n_corr, n_same_corr = \
            CorrelationValidation.validate_data_matrix(
                data_matrix, conditions, 
                n_trial = args.n_trial, 
                noise_ratio_step=args.noise_ratio_step)
            
    scores = CorrelationValidation.parse_result(result,args.p_value_threshold)
    
    print("Correlation validation result of input " + input_csv + " : ")
    print("Compendium size (#features, #samples) : " + str(data_matrix.shape))
    print("#pairs (All samples) : " + str(n_corr))
    print("#pairs (Same condition samples) : " + str(n_same_corr))
    
    print("Average score of " + str(args.n_trial) + " trials : " + 
          "{:.3f}".format(np.mean(scores.loc[:,'Score'])) + " +/- " + 
          "{:.3f}".format(np.std(scores.loc[:,'Score'])))
            
    if args.output_pvalue_table_path is not None:
        result.to_csv(args.output_pvalue_table_path)
        
    if args.output_score_path is not None:
        scores.to_csv(args.output_score_path)
        
    
    if args.plot_overall_result_path is not None:
        CorrelationValidation.plot_result(result, 
                                          args.plot_overall_result_path, 
                                          format=args.plot_saving_format)
        
    if args.plot_single_trial_result_path is not None:
        CorrelationValidation.plot_avg_corr(result, scores, 
                                            avg_corr,avg_same_corr,
                                            std_corr,std_same_corr,\
                      n_corr, n_same_corr, args.trial_idx, 
                      args.plot_single_trial_result_path, 
                      format=args.plot_saving_format)