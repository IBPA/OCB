# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:43:25 2021

PublishedDatasetComparison.py
Evaluate the quality of an omics compendium by comparing with published data directly.
Samplewise Pearson Correlation Coefficient (PCC) will be evaluated.

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

class PublishedDataComparison:
    @staticmethod
    def data_comparison(data_matrix_pd, 
                        published_data_matrix_pd, 
                        file_path,
                        xlabel = "Published Data Value (log2 transformed)", 
                        ylabel = "Input Data Value (log2 transformed)",
                        title = "Comparison with published dataset",
                        **kwargs):
        """
        
        Evaluate the quality of the input compendium by comparing published
        data.
        
        This method simply evaluate the samplewise PCC between the samples in
        input compendium and input dataset.
        
        The features in input compendium and published data can be different. 
        However, there should be at least two common features in input 
        compendium and published data and the samples listed in published data 
        should be included in input compendium.
        
        Multiple samples in input compendium mapping to one sample in published 
        data is allowed. For this case, please edit the sample label in 
        published data by concatenating two or more sample labels in input 
        compendium with '|' symbol.

        Parameters
        ----------
        data_matrix_pd : A pandas DataFrame
            Input omics compendium.
            Rows represent features and Columns represent samples.
        published_data_matrix_pd : A pandas DataFrame
            Published data.
            Rows represent features and Columns represent samples.
            
            Check the method description for more information of the criterion
            of the format of input compendium and published data
        filename : A String
            A string that specify the image file path to be saved.
            No file will be saved if it is None
        xlabel : A String, optional
            The label of x axis of the plot. The default is 
            "Published Data Value (log2 transformed)".
        ylabel : A String, optional
            The label of y axis of the plot. The default is 
            "Input Data Value (log2 transformed)".
        title : A String, optional
            The title of the plot. The default is "
            Comparison with published dataset".
        **kwargs : 
            Additional arguments for matplotlib.pyplot.savefig()

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        corr : A pandas dataframe
            The score matrix has one column 'corr' that records the average
            samplewise PCC between input compendium and published data
            of trials.
        data_matrix_for_comparison_pd : A pandas DataFrame
            The input compendium for comparison by extracting the common
            features and samples listed in published data.
            Rows represent features and Columns represent samples.
        published_data_matrix_pd : A pandas DataFrame
            The published data for comparison by extracting the common
            features and samples listed in the input compendium.
            Rows represent features and Columns represent samples.

        """
        
        MatrixUtils.check_input_pd_dataframe(
            data_matrix_pd, "input compendium", "first argument")
        
        if len(set(data_matrix_pd.index)) != len(data_matrix_pd.index):
            raise ValueError("Input compendium contains duplicated features!\n" + 
                             Utils.generate_additional_description('first argument'))
            
        if len(set(data_matrix_pd.columns)) != len(data_matrix_pd.columns):
            raise ValueError("Input compendium contains duplicated samples!\n" + 
                             Utils.generate_additional_description('first argument'))
            
        MatrixUtils.check_input_pd_dataframe(
            published_data_matrix_pd, "published data", "second argument")
            
        if len(set(published_data_matrix_pd.index)) != len(published_data_matrix_pd.index):
            raise ValueError("Published data contains duplicated features!\n" + 
                             Utils.generate_additional_description('second argument'))
            
        if len(set(published_data_matrix_pd.columns)) != len(published_data_matrix_pd.columns):
            raise ValueError("Published data contains duplicated samples!\n" + 
                             Utils.generate_additional_description('second argument'))
        
        
        features = published_data_matrix_pd.index.intersection(data_matrix_pd.index)
        
        if len(features) < 2:
            raise ValueError("Too few Common features found ( < 2) in input compendium and published data!\n" + 
                             Utils.generate_additional_description("first and second argument"))
        
        data_matrix_for_comparison_pd = pd.DataFrame(0, 
                                                  index = features,
                                                  columns = published_data_matrix_pd.columns)
        
        for col in published_data_matrix_pd.columns:
            sample_ids = str(col).split("|")
            
            for sample in sample_ids:
                if sample not in data_matrix_pd.columns:
                    raise ValueError("The sample " + sample + " is not found in input compendium.\n" + 
                                     Utils.generate_additional_description("first argument"))
            
            data_matrix_selected_samples_pd = data_matrix_pd.loc[features, sample_ids]
            if data_matrix_selected_samples_pd.shape[1] > 1:
                print("\nNotice: More than one samples in input compendium mapped to one sample in published data\n")
                print("Samples :")
                for sample in sample_ids:
                    print(sample)
                data_matrix_for_comparison_pd.loc[:, col] = np.mean(data_matrix_selected_samples_pd,axis=1)
            else:
                data_matrix_for_comparison_pd.loc[:, col] = data_matrix_selected_samples_pd
                  
            
        corr = pd.DataFrame(0, index=published_data_matrix_pd.columns, columns=['corr'])
        for col in published_data_matrix_pd.columns:
            corr.loc[col,'corr'] = np.corrcoef(published_data_matrix_pd.loc[features,col].values,
                                      data_matrix_for_comparison_pd.loc[features,col].values)[1][0]
            
        published_data_matrix_pd = np.log2(published_data_matrix_pd + 1)
        data_matrix_for_comparison_pd = np.log2(data_matrix_for_comparison_pd + 1)
        
        fig = plt.figure(dpi = 200)
        ax = fig.add_axes([0,0,1,1])
        
        xmin = np.min(np.min((published_data_matrix_pd)))
        xmax = np.max(np.max((published_data_matrix_pd)))
        
        ymin = np.min(np.min((data_matrix_for_comparison_pd)))
        ymax = np.max(np.max((data_matrix_for_comparison_pd)))
        
        axis_min = np.min([xmin,ymin])
        axis_max = np.max([xmax,ymax])
        
        ax.set_xlim([axis_min,axis_max])
        ax.set_ylim([axis_min,axis_max])
        
        title += '\nN = ' + str(published_data_matrix_pd.shape[1]) + \
                 ', Average PCC = ' + str(np.round(np.mean(corr)[0],3)) + \
                     ' ± ' + str(np.round(np.std(corr)[0],3))
                 
        ax.set_title(str(title))
        ax.set_xlabel(str(xlabel))
        ax.set_ylabel(str(ylabel))
        
        for col in published_data_matrix_pd.columns:
            ax.plot(published_data_matrix_pd.loc[features,col].values, 
                    data_matrix_for_comparison_pd.loc[features,col].values,
                    linestyle="",marker = ".")
            
            
        if file_path is not None:
            plt.savefig(file_path,bbox_inches='tight', pad_inches=0.1,  **kwargs)
            
        plt.close(fig)
        
        return corr, data_matrix_for_comparison_pd, published_data_matrix_pd
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     "Evaluate the quality of an omics\n" + 
                                     "compendium by evaluating the samplewise PCC between values\n" + 
                                     "in input compendium and published data")
    
    parser.add_argument('input_compendium_path', type=str,
                        help='The input path of the omics compendium.\n' + 
                             'Rows represent features and Columns represent samples.')
    
    parser.add_argument('published_data_path', type=str,
                        help='The input path of the published data.\n' + 
                             'Rows represent features and Columns represent samples.\n' + 
                             'The features in input compendium and published data can be different.\n' + 
                             'However, there should be at least two common features in input compendium and published data.\n' + 
                             'The samples listed in published data should be included in input compendium.\n' + 
                             '\nMultiple samples in input compendium mapping to one sample in published data is allowed.\n' + 
                             'For this case, please edit the sample label in published data by concatenating' + 
                             "two or more sample labels in input compendium with '|' symbol.")
    
    parser.add_argument('--plot_result_path', type=str, default = None,
                        help='The path of the plot of values in input compendium and published data.\n' + 
                        'No figure will be plotted if it is not provided.')
    parser.add_argument('--plot_saving_format', type=str, default = 'png',
                        help='The file format of the saving plot.\n' + 
                        'Default is png file.')
    
    args = parser.parse_args() 
    
    input_csv = args.input_compendium_path
    published_data_csv = args.published_data_path
    data_matrix_pd = pd.read_csv(input_csv,index_col = 0)
    published_data_matrix_pd = pd.read_csv(published_data_csv,index_col = 0)
    
    corr, data_matrix_for_comparison_pd, published_data_matrix_pd = \
        PublishedDataComparison.data_comparison(data_matrix_pd, 
                                               published_data_matrix_pd,
                                               args.plot_result_path,
                                               title="Comparison with published dataset",
                                               xlabel="Published Data Value (log2 transformed)",
                                               ylabel="Comparison with published dataset (log2 transformed)",
                                               format=args.plot_saving_format)
    
    print("\nPublished Data Comparison:\ninput compendium : " + input_csv + "\npublished data : " + published_data_csv + "\n")
    print("Compendium size (#features, #samples) : " + str(data_matrix_pd.shape))
    print("Published data size (#features, #samples) : " + str(published_data_matrix_pd.shape))
    
    print("Data size for comparison (#features, #samples) : " + str(data_matrix_for_comparison_pd.shape))
    
    print("\nAverage samplewise PCC = " + 
    "{:.3f}".format(np.mean(corr)[0]) + " +/- " + "{:.3f}".format(np.std(corr)[0]))