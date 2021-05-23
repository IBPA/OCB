from transcriptomic_pipeline import *
import sys
import os
import os.path

import pandas as pd
if __name__ == '__main__':
    try:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        output_png = sys.argv[3]
    except:
        raise Exception('Usage: python unsupervised_validation_script_csv_input.py <input_gene_expression_table> <output_table_path> <output_figure_path>')
        
    
    transcriptomic_pipeline = TranscriptomicDataPreparationPipeline([],[],[])
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    
    data_matrix = pd.read_csv(input_csv, index_col = 0)
    transcriptomic_pipeline.validation_pipeline.unsupervised_validation.validate_data_from_data_matrix(data_matrix,output_csv,output_png)