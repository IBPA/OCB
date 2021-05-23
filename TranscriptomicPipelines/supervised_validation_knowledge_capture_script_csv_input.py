from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path
if __name__ == '__main__':
    try:
        input_csv = sys.argv[1]
        knowledge_capture_sample = sys.argv[2]
        knowledge_capture_gene = sys.argv[3]
        output_table_path = sys.argv[4]
        output_figure_path = sys.argv[5]
    except:
        raise Exception('Usage: python supervised_validation_corr_script_csv_input.py <input_gene_expression_table> <Selected Sample List> <Selected Gene List> <output_table_path> <output_figure_path>')
        
    
    if not os.path.isfile(knowledge_capture_sample):
        raise Exception('Error: the sample list does not exist!')
        
    if not os.path.isfile(knowledge_capture_gene):
        raise Exception('Error: the gene list does not exist!')
    
    transcriptomic_pipeline = TranscriptomicDataPreparationPipeline([],[],[])
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    
    data_matrix = pd.read_csv(input_csv, index_col = 0)
    transcriptomic_pipeline.validation_pipeline.supervised_validation.knowledge_capture_validation_from_data_matrix(data_matrix, knowledge_capture_sample, knowledge_capture_gene, output_table_path, output_figure_path)
    