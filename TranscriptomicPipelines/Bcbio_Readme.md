# Validate the gene expression table from bcbio pipeline
<h4>This example provides a step-by-step example to make a Salmonella example compendium using <a href = https://github.com/bcbio/bcbio-nextgen>bcbio</a>.</h4>

## Bcbio Installation and Configuration
There are several necessary steps to generate the gene expression profile table using bcbio pipeline.

1. Install the bcbio pipeline using the commands as follows ([See details on bcbio repository](https://github.com/bcbio/bcbio-nextgen/blob/master/README.md))
    ```shell script
    wget https://raw.githubusercontent.com/bcbio/bcbio-nextgen/master/scripts/bcbio_nextgen_install.py
    python bcbio_nextgen_install.py /usr/local/share/bcbio --tooldir=/usr/local \
    --genomes hg38 --aligners bwa --aligners bowtie2
    ```
    * It will takes about 16 hours and consume about 75 GB disk space.
    * If one of the installation steps failed, please remove all of the temporary files and directory (the directory name contains `tmp`) before run the installation script again.
    
1. Install the Salmonella LT2 genome template:
    * Please make sure you located the file `bcbio_setup_genome.py` and add this path to PATH environment variable.
    * Prepare the [reference genome sequence](https://github.com/IBPA/OCB/tree/master/TestFiles/GCA_000006945.2_ASM694v2_genomic.fna).
    * Prepare the [gene annotation file](https://github.com/IBPA/OCB/tree/master/TestFiles/LT2GenomeForBcbio.gtf).
    * Use the following commands to build <em>Salmonella enterica</em> (LT2 strain) genome ([See detail](https://bcbio-nextgen.readthedocs.io/en/latest/contents/configuration.html#adding-custom-genomes)):
    ```shell script
            bcbio_setup_genome.py \
            -f GCA_000006945.2_ASM694v2_genomic.fna \
            -g LT2GenomeForBcbio.gtf \
            -i bowtie2 \
            -n Salmonella -b LT2 --LT2_01
    ```
    * NOTE 1: The script may not execute correctly. You may need to fix the error with the following two steps. 
      * First, find the python script dexseq_prepare_annotation.py and convert it into Python3-compatible format. 
      * Second, if the error message shows that one of the transcript gtf file is empty, please copy your input gtf file to the temporary directory and run the script again.
      * Third, please install HTSeq python package inside the bcbio-python environment:
    ```shell script
            (bcbio installation path)/anaconda/bin/python -m pip install HTSeq
    ```  
      
    * NOTE 2: The provided gtf file is simplfied by removing additional description. In addition, all "CDS" term are converted into "exon".

1. Download the sample RNA-seq files and convert it into fastq.gz format. Please make sure [sratoolkit](https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit) is installed:
    ```
    mkdir seqc/input
    cd seqc/input
    prefetch SRR1585217
    prefetch SRR1585225
    prefetch ERR1706735
    prefetch ERR1706741
    
    fastq-dump --gzip --split-3 SRR1585217/SRR1585217.sra 
    fastq-dump --gzip --split-3 SRR1585225/SRR1585225.sra 
    fastq-dump --gzip --split-3 ERR1706735/ERR1706735.sra 
    fastq-dump --gzip --split-3 ERR1706741/ERR1706741.sra 
    ```

1. Configure and test bcbio installation. The following shows the basic configuration such as running in the local single node. For advanced configuration, please refer [bcbio official document](https://bcbio-nextgen.readthedocs.io/en/latest/contents/bulk_rnaseq.html).
   
   1. Download a template yaml file for RNA-seq analysis in bcbio.
       ```
       wget --no-check-certificate https://raw.githubusercontent.com/bcbio/bcbio-nextgen/master/config/examples/rnaseq-seqc.yaml
       ```
       Modify the template yaml file so that it can perform RNA-seq analysis on Salmonella using bowtie2:
       
       ```
       details:
          - analysis: RNA-seq
            genome_build: LT2
            algorithm:
              quality_format: standard
              aligner: bowtie2
              strandedness: unstranded
            upload:
              dir: ../final
            resources:
              star:
                cores: 4
                memory: 2G
       ```
   
    1. Prepare the sample description file and name it as `seqc.csv` as follows:
       ```
       samplename,description,panel
       SRR1585217, Test1, Test1
       SRR1585225, Test2, Test2
       ERR1706735, Test3, Test3
       ERR1706741, Test4, Test4
       ```
   
    1. Create the project and run the analysis (rename the following .yaml and .csv name if necessary):
       ```
       bcbio_nextgen.py -w template rnaseq-seqc.yaml seqc.csv seqc/input/*.gz
       ```
    
    1. Run the test case analysis. 
       ```
       cd seqc/work
       bcbio_nextgen.py ../config/seqc.yaml -n 8
       ```
       The gene expression count table will be stored in `seqc/final/(date)_seqc/counts/tximport-counts.csv`
       
1. Run the entire Salmonella example samples with this [sample description file](https://github.com/IBPA/OCB/tree/master/TestFiles/SampleTableForBcbio.csv). To download and convert the RNA-seq files, please use the following scripts:
   * [Prefetch script](https://github.com/IBPA/OCB/tree/master/TestFiles/DownloadAllSRASamples.sh)
   * [Fastq-dump script](https://github.com/IBPA/OCB/tree/master/TestFiles/FastqDumpAllSamples.sh)

1. Please refer the validating part in the [step-by-step tutorial](https://github.com/IBPA/OCB/tree/master/TranscriptomicPipelines#3-validation)
