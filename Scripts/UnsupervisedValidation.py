# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:46:33 2021

UnsupervisedValidation.py: 
Evaluate the quality of an omics compendium without any metadata.

This approach assumes that an imputation algorithm can impute the missing value
with low error if the quality of the compendium is good. In addition, a good 
compendium should have high tolerance of noise.

By evaluating the imputation error of the compendium with different level of 
perturbation (i.e. mixing with noise with different noise ratio), this approach
can evaluate and define the quality of the compendium.


@author: ChengEn Tan (cetan@ucdavis.edu)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from Utils import *
import missingpy
from missingpy import MissForest, KNNImputer

import argparse

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DropRatioUtils:
    '''
    Utilities for checking the drop ratio input
    
    Methods in this class will be called internally for checking the input
    '''
    
    @staticmethod
    def check(drop_ratio, data_matrix, argument_order):
        DropRatioUtils.check_type(drop_ratio, argument_order)
        DropRatioUtils.check_too_low(drop_ratio, data_matrix, argument_order)
        DropRatioUtils.check_too_high(drop_ratio, argument_order)

    @staticmethod
    def check_type(drop_ratio, argument_order):
        if type(drop_ratio) is not float and type(drop_ratio) is not int:
            raise TypeError(
                "The drop ratio is not a number!\n" + 
                Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def check_too_low(drop_ratio, data_matrix, argument_order):
        if np.round(drop_ratio * data_matrix.shape[0]) <= 0:
            raise ValueError(
                "The drop ratio is too low or the input matrix may be too small:\n" + 
                "The number of drop values is less than or equal to zero!\n" + 
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def check_too_high(drop_ratio, argument_order):
        if drop_ratio > 0.5:
            raise ValueError(
                "The drop ratio is too high:\n" + 
                "The number of drop values is high than 0.5!\n" + 
                Utils.generate_additional_description(argument_order))
                
class DroppedIndicesUtils:
    '''
    Utilties for checking the dropped indices
    
    Methods in this class will be called internally for checking the input
    '''
    @staticmethod
    def check(dropped_indices, input_matrix, argument_order):
        DroppedIndicesUtils.check_type(
            dropped_indices, argument_order)
        DroppedIndicesUtils.check_len(
            dropped_indices, input_matrix, argument_order)
        
        for element in dropped_indices:
            DroppedIndicesUtils.check_element_type(element, argument_order)
            DroppedIndicesUtils.check_element_shape(element, argument_order)
            DroppedIndicesUtils.check_element_empty(element, argument_order)
            DroppedIndicesUtils.check_element_too_long(element, input_matrix, argument_order)
            DroppedIndicesUtils.check_element_duplicated(element, argument_order)
            DroppedIndicesUtils.check_sub_element_type(element, argument_order)
            
            for sub_element in element:
                DroppedIndicesUtils.check_sub_element_negative(
                    sub_element, argument_order)
                DroppedIndicesUtils.check_sub_element_out_of_bound(
                    sub_element, input_matrix, argument_order)

    @staticmethod
    def check_type(dropped_indices, argument_order):
        if type(dropped_indices) is not list:
            raise TypeError("The indices list of dropped values is not a list!\n" + 
                            Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def check_len(dropped_indices, input_matrix, argument_order):
        if len(dropped_indices) != input_matrix.shape[1]:
            raise ValueError(
                "The length of the indices list of dropped values is not equal to the #column of matrix before dropping values\n" + 
                Utils.generate_additional_description(argument_order))
                
    @staticmethod
    def check_element_type(element, argument_order):
        if type(element) is not np.ndarray:
            raise TypeError(
                "At least one element of the indices list of dropped values is not a numpy array!\n" + 
                        Utils.generate_additional_description(argument_order))
                        
    @staticmethod
    def check_element_shape(element, argument_order):
        if len(element.shape) != 1:
            raise TypeError(
                "At least one element of the indices list of dropped values is not a 1D numpy array!\n" + 
                        Utils.generate_additional_description(argument_order))
        
    @staticmethod
    def check_element_empty(element, argument_order):
        if element.shape[0] == 0:
            raise ValueError(
                "At least one element of the indices list of dropped values is empty!\n" + 
                        Utils.generate_additional_description(argument_order))
        
    @staticmethod
    def check_element_too_long(element, input_matrix, argument_order):
        if element.shape[0] > input_matrix.shape[0]:
            raise ValueError(
                "At least one element of the indices list of dropped values is longer than the #row of imputed matrix!\n" + 
                        Utils.generate_additional_description(argument_order))
        
    @staticmethod
    def check_element_duplicated(element, argument_order):
        if len(element) > len(set(element)):
            raise ValueError(
                "At least one element of the indices list of dropped values contains duplicated values!\n" + 
                        Utils.generate_additional_description(argument_order))
        
    @staticmethod
    def check_sub_element_type(element, argument_order):
        try:
            if np.isnan(np.sum(element)):
                raise ValueError(
                    "At least one element of the indices list of dropped "+ 
                    "values contains NaN values!\n" + 
                            Utils.generate_additional_description(argument_order))
            if np.isinf(np.sum(element)):
                raise ValueError(
                    "At least one element of the indices list of dropped values contains infinite values!\n" + 
                            Utils.generate_additional_description(argument_order))
        except TypeError:
            raise ValueError(
                    "At least one element of the indices list of dropped values contains non numerical values!\n" + 
                            Utils.generate_additional_description(argument_order))
    
    @staticmethod
    def check_sub_element_negative(sub_element, argument_order):
        if sub_element < 0:
            raise ValueError(
                "At least one element of the indices list of dropped values contains negative values!\n" + 
                        Utils.generate_additional_description(argument_order))
                        
    @staticmethod
    def check_sub_element_out_of_bound(sub_element, input_matrix, argument_order):
        if sub_element >= input_matrix.shape[0]:
            raise ValueError(
                "At least one element of the indices list of dropped values contains a value greater than #row of imputed matrix!\n" + 
                        Utils.generate_additional_description(argument_order))

class ImputerUtils:
    @staticmethod
    def check(imputer, argument_order):
        if type(imputer) is not missingpy.knnimpute.KNNImputer \
            and type(imputer) is not missingpy.missforest.MissForest:
            raise TypeError("The type of the imputer is invalid.\n" + 
                            "The imputer should be missingpy.knnimpute.KNNImputer or" +
                            "missingpy.missforest.MissForest.\n" + 
                            Utils.generate_additional_description(argument_order)
                            )
            

class UnsupervisedValidation:
    @staticmethod
    def drop_values(data_matrix : np.ndarray, drop_ratio : float):
        """
        Randomly drop the values in the data matrix by marking them as NAN
        
        This method drops values by columns. The missing value ratio for each
        row can be different.
        
        It returns the data matrix after dropping values and the indices of
        dropped values for each column.

        Parameters
        ----------
        data_matrix : A 2D numpy array
            The input matrix. 
            Matrices with NaN value and infinite values are not allowed.
        drop_ratio : A float
            A number between (1/#row of input matrix) to 0.5. 
            0.1 means 10% values will be dropped for each column of perturbed matrix. 
            At least one value should be dropped for each column of perturbed matrix.

        Returns
        -------
        dropped_matrix : A 2D numpy array
            The matrix after dropping values
        dropped_indices : A list of 1D numpy array
            A list with the indices of dropped values as 1D numpy array

        """
        
        """Check the first argument
        """
        MatrixUtils.check_input_matrix(data_matrix, "input data matrix","first argument")
        
            
        """Check the second argument
        """
        DropRatioUtils.check(drop_ratio, data_matrix, "second argument")
        
        dropped_matrix = np.copy(data_matrix)
        dropped_indices = []
        n_dropped_each_col = np.round(
                        drop_ratio * dropped_matrix.shape[0]).astype(int)
        for colidx in range(dropped_matrix.shape[1]):
            dropped_indices_this_col = np.random.choice(
                dropped_matrix.shape[0], n_dropped_each_col, replace=False)
            dropped_matrix[dropped_indices_this_col, colidx] = np.nan
            dropped_indices.append(dropped_indices_this_col)
        
        return dropped_matrix, dropped_indices
    
    @staticmethod 
    def evaluate_impute_error(original_matrix : np.ndarray, 
                              imputed_matrix : np.ndarray, 
                              dropped_indices : list):
        """
        Evaluate average imputation error
        
        This method evaluates average imputation error of the entire data
        matrix given the data matrix before dropping value, imputed data
        matrix after dropping values, and the indices of dropped value

        Parameters
        ----------
        original_matrix : A 2D numpy array
            The data matrix before dropping values. 
            Matrices with NaN value and infinite values are not allowed.
        imputed_matrix : A 2D numpy array
            The imputed data matrix.
            Matrices with NaN value and infinite values are not allowed.
        dropped_indices : A list of 1D numpy array
            The list of the indices of dropping values 

        Raises
        ------
        ValueError
            Raise when the shape of data_matrix and noise_ratrix are different.

        Returns
        -------
        error : A float (np.float)
            The average imputation error of all imputed values in the matrix

        """

        """check the matrices
        """
        MatrixUtils.check_input_matrix(original_matrix,"matrix before dropping values",
                           "first argument")
        MatrixUtils.check_input_matrix(imputed_matrix,"imputed matrix","second argument")
        
        """Size of the two matrices should be the same
        """
        MatrixUtils.check_two_matrix_shape_consistent(
            original_matrix, imputed_matrix, 
            "matrix before dropping values", 
            "the imputed matrix", 
            "first two arguments")
        
        
        """check the indices of dropped values
        """
        DroppedIndicesUtils.check(dropped_indices, original_matrix, "third argument")
        
        truth_values = []
        imputed_values = []
        for colidx in range(original_matrix.shape[1]):
            truth_values.extend(
                original_matrix[dropped_indices[colidx], colidx].tolist())
            imputed_values.extend(
                imputed_matrix[dropped_indices[colidx], colidx].tolist())
        
        truth_values = np.array(truth_values)
        imputed_values = np.array(imputed_values)
        error = np.mean(np.abs(imputed_values - truth_values))
        
        return error
    
    
    @staticmethod
    def validate_data_matrix_specific_noise_ratio(imputer,
                                                  data_matrix,
                                                  noise_matrix,
                                                  noise_ratio,
                                                  drop_ratio = 0.5):
        """
        Run unsupervised validation given data matrix given inputer, data, noise, 
        specified noise ratio, and specified drop ratio.
        
        The input data matrix will be mixed with the noise with specified noise ratio
        to generate perturbed matrix, and then some values will be dropped with 
        specified noise ratio, finally a missing value imputation algorithm will be 
        applied to impute these values. The location of dropped values will be recorded 
        so that the imputation error can be evaluated.

        Parameters
        ----------
        imputer : missingpy.knnimpute.KNNImputer or missingpy.missforest.MissForest.
            The imputer in missingpy package. It can be two different
            imputer: missingpy.knnimpute.KNNImputer or 
            missingpy.missforest.MissForest.
        data_matrix : A 2D numpy array
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
        noise_matrix : A 2D numpy array
            A noise matrix which can be generated by Utils.prepare_noise() in 
            this module by randomly permute the input data matrix.
            Matrices with NaN value and infinite values are not allowed.
        noise_ratio : A float
            A number between 0.0 to 1.0, 0.0 means no noise, 
            0.5 means linear combination of 50% original data matrix and 50% 
            noise, and 1 means 100% noise.
        drop_ratio : A float, optional
            A number between (1/#row of input matrix) to 0.5. 
            0.1 means 10% values will be dropped for each column of perturbed matrix. 
            At least one value should be dropped for each column of perturbed matrix.
            The default is 0.5.

        Raises
        ------
        ValueError
            Raise when the shape of data_matrix and noise_ratrix are different.

        Returns
        -------
        error : A float (np.float)
            The average imputation error of all imputed values in the matrix.
        mixed_matrix : A 2D numpy array
            The perturbed data matrix.
        imputed_matrix : A 2D numpy array
            The perturbed data matrix with dropping values and missing
            value imputation.
        dropped_indices : A list of 1D numpy array
            The location of dropped values of the matrix.

        """
         
        
        """check the imputer
        """
        ImputerUtils.check(imputer, "first argument")

        """check the matrices
        """
        MatrixUtils.check_input_matrix(data_matrix,"input data matrix",
                           "second argument")
        MatrixUtils.check_input_matrix(noise_matrix,"noise matrix","third argument")
        
        """Size of the two matrices should be the same
        """
        if data_matrix.shape != noise_matrix.shape:
            raise ValueError(
                "The sizes of input data matrix and noise matrix are different!\n" + 
                Utils.generate_additional_description("2nd and 3rd arguments"))
                
                
        """Check the fourth argument
        """
        NoiseRatioUtils.check_noise_ratio(noise_ratio, "fourth argument")
        
                
        """Check the fifth argument
        """
        DropRatioUtils.check(drop_ratio, data_matrix, "fifth argument")
                
        
        mixed_matrix = noise_matrix * noise_ratio + \
                       data_matrix * (1 - noise_ratio)
                       
        
        mixed_matrix_t = mixed_matrix.transpose()
        dropped_matrix_t, dropped_indices = UnsupervisedValidation.drop_values(mixed_matrix_t, drop_ratio)
        
        imputed_matrix_t = imputer.fit_transform(dropped_matrix_t)
        imputed_matrix = imputed_matrix_t.transpose()
            
        error = UnsupervisedValidation.evaluate_impute_error(mixed_matrix_t, imputed_matrix_t, dropped_indices)
        return error, mixed_matrix, imputed_matrix, dropped_indices
    
    @staticmethod
    def validate_data_matrix(imputer,
                             data_matrix, 
                             n_trial = 100, 
                             noise_ratio_step = 0.1, 
                             drop_ratio = 0.5):
        """
        
        Run unsupervised validation given data matrix given imputer, data, #trial, 
        step of noise ratio, and specified drop ratio.
        
        This method is the wrapper of validate_data_matrix_specific_noise_ratio(). It 
        will run that function n_trial times with different noise ratios from 0 to 1
        with specified noise ratio step. The noise matrix will be generated for each 
        method call.
        

        Parameters
        ----------
        imputer : missingpy.knnimpute.KNNImputer or missingpy.missforest.MissForest.
            The imputer in missingpy package. It can be two different
            imputer: missingpy.knnimpute.KNNImputer or 
            missingpy.missforest.MissForest.
        data_matrix : A 2D numpy array
            The compendium for quality evaluation.
            Rows represent feature and Columns represent samples.
            Matrices with NaN value and infinite values are not allowed.
        n_trial : An integer, optional
            An integer specify how many iterations to be run. One 
            iteration means calling validate_data_matrix_specific_noise_ratio() 
            with noise ratio from 0 to 1 with specified noise ratio step.
            The default is 100.
        noise_ratio_step : A float, optional
            A number between 0 to 0.5 (0 is not allowed)
            The default is 0.1.
        drop_ratio : A float, optional
            A number between (1/#row of input matrix) to 0.5. 
            0.1 means 10% values will be dropped for each column of perturbed matrix. 
            At least one value should be dropped for each column of perturbed matrix.
            The default is 0.5.

        Returns
        -------
        result : A pandas dataframe
            The average imputation error for each trial
            with different noise ratio. Each row represent different ratios 
            and each column represent different trials.

        """
        
        """check the imputer
        """
        ImputerUtils.check(imputer, "first argument")

        """check the matrix
        """
        MatrixUtils.check_input_matrix(data_matrix,"input data matrix",
                           "second argument")
        
        """check the ntrial argument:
        """
        NTrialUtils.check_n_trial(n_trial, "third argument")
        
        """check the noise ratio step:
        """
        NoiseRatioUtils.check_noise_ratio_step(noise_ratio_step, "fourth argument")
        
        """check drop ratio
        """
        DropRatioUtils.check(drop_ratio, data_matrix, 
                                                "fifth argument")
        
        
        noise_ratios = np.arange(0.0, 1.0 + noise_ratio_step, noise_ratio_step)
        result = pd.DataFrame(0, index = noise_ratios, columns = 
                              np.arange(n_trial))
        
        for x in np.arange(n_trial):
            noise_matrix = Utils.prepare_noise(data_matrix)
            for noise_ratio in noise_ratios:
                error, mixed_matrix, imputed_matrix, dropped_indices = \
                    UnsupervisedValidation.validate_data_matrix_specific_noise_ratio(
                        imputer, 
                        data_matrix,
                        noise_matrix,
                        noise_ratio,
                        drop_ratio)
                
                result.loc[noise_ratio,x] = error
        
        return result
    
    @staticmethod
    def read_result(result):
        """
        Parse the result and evaluate the score for each trial
        
        This method evaluate the score for each trial by evaluating the ratio 
        of the following two matrix:
        (1) The area between max error horizontal line and the error curve
            in y axis, and between the noise ratio which yields min error
            and max error in x axis. Notice that if the noise ratio yields
            max error is less than the noise ratio yields min error, this 
            area will be counted as zero.
            
        (2) The area between max error and min error in y axis, and between
            0 and 1 in x axis.
        

        Parameters
        ----------
        result : A pandas data frame
            A pandas dataframe that contains the unsupervised validation
            result generated from the method validate_data_matrix(). 
            Each row represent different ratios, and each column 
            represents different trials.
                
            The index of this dataframe should be a valid noise ratio.

        Returns
        -------
        area_table : A pandas data frame
            A pandas dataframe that contains the score for each 
            trial with the following fields:
            
            max -- the max error in this trial
            accumulate -- the area belows the error curve
            min -- the min error in this trial
            
            Since the noise ratio is between 0 to 1, area (2) is equal to
            (max - min), and area (1) is (max - accumulate),
            therefore, the score can be evaluated by 
            1 - (accumulate - min)/(max - min) or 
            (max - accumulate)/(max - min)      

        """
        
        
        """Check result
        """
        ResultUtils.check_unsupervised_validation_result(result, "first argument")
        
        x = np.array(result.index)
        area_table = pd.DataFrame(0, index = ['max','accumulate','min','score'], columns = 
                                   result.columns)
        
        for col in result.columns:
            cur_y = np.array(result[col])
            min_val = np.min(cur_y)
            max_val = np.max(cur_y)
            
            min_idx = 0
            max_idx = 0
            for i in range(len(x)):
                if np.abs(cur_y[i] - min_val) < 1e-5:
                    min_idx = i
            
            for i in range(len(x)):
                if np.abs(cur_y[i] - max_val) < 1e-5:
                    max_idx = i
                    break
                
            min_area = 0
            max_area = 0
            accumulate_area = 0
            
            not_count = False
            for i in range(len(x)-1):
                min_area += ((min_val + min_val) * (x[i+1] - x[i]) * 0.5)
                max_area += ((max_val + max_val) * (x[i+1] - x[i]) * 0.5)
                
                if i >= min_idx and (i+1) <= max_idx:
                    accumulate_area += ((cur_y[i+1] + cur_y[i]) * (x[i+1] - x[i]) * 0.5)
                else:
                    accumulate_area += ((max_val + max_val) * (x[i+1] - x[i]) * 0.5)
                
            area_table.loc['max', col] = max_area
            area_table.loc['accumulate', col] = accumulate_area
            area_table.loc['min', col] = min_area
            area_table.loc['score',col] = 1 - (accumulate_area - min_area)/(max_area - min_area)
            
        return area_table
    
    def plot_result(result, filename, **kwargs):
        """
        
        Plot the error curve of all trials

        Parameters
        ----------
        result : A pandas data frame
            A pandas dataframe that contains the unsupervised validation
            result generated from the method validate_data_matrix(). 
            Each row represent different ratios, and each column 
            represents different trials.
                
            The index of this dataframe should be a valid noise ratio.
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
        ResultUtils.check_unsupervised_validation_result(result, "first argument")
        
        
        fig = plt.figure(dpi = 200)
        unsupervised_plot = result.plot(
            title='Unsupervised validation results (' + 
            str(result.shape[1]) + 
            ' trials)',
            legend = False, 
            xlabel='Noise Ratio', ylabel='Average Absolute Error')
        
        if filename is not None:
            plt.savefig(str(filename), **kwargs)
        plt.close(fig)
        
    def plot_single_result(result, idx_trial, filename, **kwargs):
        """
        Plot the error curve of one trial with specified index

        Parameters
        ----------
        result : A pandas data frame
            A pandas dataframe that contains the unsupervised validation
            result generated from the method validate_data_matrix(). 
            Each row represent different ratios, and each column 
            represents different trials.
                
            The index of this dataframe should be a valid noise ratio.
        idx_trial : An integer
            An integer specify the trial index
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
        ResultUtils.check_unsupervised_validation_result(result, "first argument")
        
        """Check idx_trial
        """
        NTrialUtils.check_trial_index(idx_trial, result, "second argument")
        
        fig = plt.figure(dpi = 200)
        unsupervised_plot = result.iloc[:,idx_trial].plot(
            title='Unsupervised validation results',
            legend = False, 
            xlabel='Noise Ratio', ylabel='Average Absolute Error',
            color='black')
        min_val = np.min(result.iloc[:,idx_trial])
        max_val = np.max(result.iloc[:,idx_trial])
        
        unsupervised_plot.plot(np.array([0, 1]),np.array([min_val, min_val]),'r--')
        unsupervised_plot.plot(np.array([0, 1]),np.array([max_val, max_val]),'r--')
        
        min_idx = np.where(result.iloc[:,idx_trial] == np.min(result.iloc[:,idx_trial]))[0][0]
        max_idx = np.where(result.iloc[:,idx_trial] == np.max(result.iloc[:,idx_trial]))[0][0]
        
        
        background_color = plt.get_cmap()(1.0)
        background_color = (background_color[0],background_color[1],background_color[2],0.5)
        unsupervised_plot.fill_between([0,1], 
                                       min_val,
                                       max_val,
                                       color = background_color)
        
        if min_idx < max_idx:
            foreground_color = plt.get_cmap()(0.5)
            foreground_color = (foreground_color[0], foreground_color[1], foreground_color[2],0.7)
            unsupervised_plot.fill_between(result.iloc[(min_idx):(max_idx+1),idx_trial].index, 
                                           result.iloc[(min_idx):(max_idx+1),idx_trial],
                                           max_val,
                                           color = foreground_color)
        if filename is not None:
            plt.savefig(str(filename), **kwargs)
        plt.close(fig)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     "Evaluate the quality of an omics\n" + 
                                     "compendium using an unsupervised approach")
    
    parser.add_argument('input_compendium_path', type=str,
                        help='The input path of the omics compendium.\n' + 
                             'Rows represent features and Columns represent samples.')
    parser.add_argument('--imputer', choices = ['knn','missForest'], default = 'knn',
                        help='The option of imputation approach. Can be knn or missForest.')
    parser.add_argument('--n_trial', type=int, default = 100,
                        help='The number of trial.')
    parser.add_argument('--noise_ratio_step', type=float, default = 0.01,
                        help='Noise ratio step. The validation procedure will evaluate' + 
                        'the imputation error of different ratio from 0 to 1 with' + 
                        'this specified noise ratio step.')
    parser.add_argument('--drop_ratio', type=float, default = 0.5,
                        help='The drop ratio. For each column, the specified' +
                             'ratio of values will be dropped and imputed later.\n')
    parser.add_argument('--output_impute_error_table_path', type=str, default = None,
                        help='The path of the output impute error table (in csv format).\n' + 
                        'No file will be produced if it is not provided.')
    parser.add_argument('--output_impute_area_table_path', type=str, default = None,
                        help='The path of the output area table (in csv format).\n' + 
                        'No file will be produced if it is not provided.')                    
    parser.add_argument('--plot_overall_result_path', type=str, default = None,
                        help='The path of the plot of overall result.\n' + 
                        'No figure will be plotted if it is not provided.')
    parser.add_argument('--plot_saving_format', type=str, default = 'svg',
                        help='The file format of the saving plot.\n' + 
                        'Default is svg file.')

    parser.add_argument('--plot_single_trial_result_path', type=str, default = None,
                        help='The path of the plot of a single trial result.\n' + 
                        'No figure will be plotted if it is not provided.')
    parser.add_argument('--trial_idx', type=int, default = 0,
                        help='The trial index for plotting.\n' + 
                        'Default is 0 (i.e. the first trial result will be plotted).')
    
    parser.add_argument('--demo', action = 'store_true',
                        help='To run the demo. The imputation ' + 
                             'procedure will be skipped and use the demo' + 
                             'dataset for demostration.')                    
    
    args = parser.parse_args()  
    
    input_csv = args.input_compendium_path
    if args.imputer == 'knn':
        imputer = KNNImputer()
    elif args.imputer == 'missForest':
        imputer = MissForest()
    
    data_matrix = np.array(pd.read_csv(input_csv, index_col = 0))    
    if args.demo == True:
        print("Running Demo Mode!")
        nrow = max(int(np.ceil(data_matrix.shape[0] * 0.1)),min(200,data_matrix.shape[0]))
        ncol = max(int(np.ceil(data_matrix.shape[1] * 0.1)),min(10,data_matrix.shape[1]))
        args.n_trial = 3
        args.noise_ratio_step = 0.25
        args.drop_ratio = 0.1
        data_matrix = data_matrix[0:nrow,0:ncol]

    result = UnsupervisedValidation.validate_data_matrix(
        imputer, data_matrix, n_trial = args.n_trial, 
        noise_ratio_step=args.noise_ratio_step,
        drop_ratio=args.drop_ratio)
    
    area_table = UnsupervisedValidation.read_result(result)
    
    print("Unsuperivsed validation result of input " + input_csv + " : ")
    print("Compendium size (#features, #samples) : " + str(data_matrix.shape))
    print("Average score of " + str(args.n_trial) + " trials : " + 
          "{:.3f}".format(np.mean(area_table.loc['score',:])) + " +/- " + 
          "{:.3f}".format(np.std(area_table.loc['score',:])))
    if args.output_impute_error_table_path is not None:
        result.to_csv(args.output_impute_error_table_path)
    if args.output_impute_area_table_path is not None:
        area_table.to_csv(args.output_impute_area_table_path)
    
    if args.plot_overall_result_path is not None:
        UnsupervisedValidation.plot_result(
                    result, 
                    args.plot_overall_result_path, 
                    format=args.plot_saving_format)
        
    if args.plot_single_trial_result_path is not None:
        UnsupervisedValidation.plot_single_result(
                    result, 
                    args.trial_idx, 
                    args.plot_single_trial_result_path, 
                    format=args.plot_saving_format)
