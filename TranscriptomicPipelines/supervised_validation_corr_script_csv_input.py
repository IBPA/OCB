from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path
if __name__ == '__main__':
    try:
        input_csv = sys.argv[1]
        input_corr_path = sys.argv[2]
        output_table_path = sys.argv[3]
        output_figure_path = sys.argv[4]
    except:
        raise Exception('Usage: python supervised_validation_corr_script_csv_input.py <input_gene_expression_table> <study-condition-sample_mapping_table> <output_table_path> <output_figure_path>')
        
    if not os.path.isfile(input_corr_path):
        raise Exception('Error: the study-condition-sample_mapping_table does not exist!')
    
    transcriptomic_pipeline = TranscriptomicDataPreparationPipeline([],[],[])
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    
    data_matrix = pd.read_csv(input_csv, index_col = 0)
    transcriptomic_pipeline.validation_pipeline.supervised_validation.correlation_validation_from_data_matrix(data_matrix, input_corr_path, output_table_path, output_figure_path)