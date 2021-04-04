# Omics Compendium Builder (OCB): An automated omics compendium preparation pipeline

This toolkit can prepare the transcriptomic compendium, a normalized, format-consistent data matrix across samples from different studies, by collecting the samples in <a href="https://www.ncbi.nlm.nih.gov/sra">Sequencing Read Archive (SRA)</a> database given the topic you are interested in and your target species.

![Figure 1. The entire transcriptomic compendium pipeline](./images/Figure1.png)
**Figure 1. The entire transcriptomic compendium pipeline.** The process consists of 6 steps: **1**, Metadata preparation by extracting run information from SRA. **2**, Downloading sequencing data in FASTA format. **3**, Aligning sequences with reference genomes. **4**, Generating gene expression profile for each run given the corresponding sequence direction information (BED) and gene annotation. **5**, Normalizing gene expression profile table. **6**, Different approaches for validating the quality of the generated compendium.

# Directories
- [`TranscriptomicPipelines`](./TranscriptomicPipelines): The folder contains the source code for OCB.

# Getting Started

Download the entire repository:
```
git clone https://github.com/IBPA/OCB.git
```

## Dependencies

<!-- TODO after CHengEn's setup script is done. -->
### Software

Make sure the following softwares are installed.
```
python==3.6
sra-tools==2.10.8
bowtie==2.3.4
```

### Packages

Make sure to install the following Python packages.
```
biopython==1.74
pandas==0.25
RSeQC==3.0.0
HTSeq==0.11.2
missingpy==0.2.0
scikit-learn==0.20.1
matplotlib==3.0.2
```

## Running

The pipeline consists of two components: Compendium construction and validation. The pipeline builds a compendium using the sample lists and gene annotations provided by users. Then it provides different validation approaches to validate the statistical siginificance and usefulness of the generated compendiums. For more detailed usage, see this [step-by-step tutorial](./TranscriptomicPipelines/README.md).

### Constructing Compendium

#### Input

In order to build a compendium, the script needs three input arguments:
- The path to a sample list file ([Example](./TestFiles/SalmonellaExampleSampleList.csv))
- The path to a gene annotation file.
- An output compendium name.

#### Output

This script will generate a directory with specified compendium name and many files in the directory. There are two outputs that are the most important:
- Normalized data matrix: A CSV table that contains normalized gene expression profiles of all samples. Each row represents different genes and each column represents different samples. The output is stored in '($compendium_name)_NormalizedDataMatrix.csv'.
- Compendium in binary format: A python object that contains the normalized gene expression table and the recorded parameters. It can be used for optional validation. The output is stored in '($compendium_name)_projectfile.bin'.


#### Example
```
cd TranscriptomicPipelines
python build_compendium_script.py \
    ../TestFiles/SalmonellaExampleSampleList.csv \
    ../TestFiles/GCF_000006945.2_ASM694v2 \
    SalmonellaExample
```

### Validating Compendium

The pipeline provides several approaches to ensure the quality of the generated compendiums:
- [Unsupervised validation](./TranscriptomicPipeline/VALIDATION.md)
- [Supervised validation with correlation](./TranscriptomicPipeline/VALIDATION.md#an-supervised-approach----correlation-validation)
- [Supervised validation with knowledge capture](./TranscriptomicPipeline/VALIDATION.md#an-supervised-approach----knowledge-capture-validation)
- [Supervised validation with published data](./TranscriptomicPipeline/VALIDATION.md#an-supervised-approach----published-data-comparison)

Please refer to [validation totorial](./TranscriptomicPipeline/VALIDATION.md).

# Authors

- [ChengEn Tan](https://github.com/bigghost2054) as the project lead, main author, and the main developer.
- [Fangzhou Li](https://github.com/fangzhouli) as the metadata pipeline developer and the code reviewer.
- [Dr. Minseung Kim](https://github.com/minseven) as the technical advisor.
- [Dr. Ilias Tagkopoulos](https://github.com/itagkopoulos) as the project supervisor and advisor.

<!-- TODO if we want to list more detailed contact info like ChengEn's. -->
# Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

<!-- # Citation

TODO -->

# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](./LICENSE) file for details.

<!-- If we have any. -->
<!-- # Acknowledgements

TODO -->

<!-- # References

<ol>
	<li>Langmead, B. & Salzberg, S. L. Fast gapped-read alignment with Bowtie 2. Nat Methods 9, 357–9 (2012).</li>
	<li>Anders, S., Pyl, P. T. & Huber, W. HTSeq--a Python framework to work with high-throughput sequencing data. Bioinformatics 31, 166–9 (2015).</li>
	<li>Guimera, R. V. bcbio-nextgen: Automated, distributed next-gen sequencing pipeline. EMBnet. journal 17, 30 (2011).</li>
	<li>Stein, L. Generic feature format version 3 (GFF3). Seq. Ontol. Proj 1, (2013).</li>
	<li>Cock, P. J. A., Fields, C. J., Goto, N., Heuer, M. L. & Rice, P. M. The Sanger FASTQ file format for sequences with quality scores, and the Solexa/Illumina FASTQ variants. Nucleic Acids Res 38, 1767–1771 (2010).</li>
	<li>Kodama, Y., Shumway, M. & Leinonen, R. The Sequence Read Archive: explosive growth of sequencing data. Nucleic acids research 40, D54–D56 (2011).</li>
	<li>Cock, P. J. et al. Biopython: freely available Python tools for computational molecular biology and bioinformatics. Bioinformatics 25, 1422–1423 (2009).</li>
	<li>Lipman, D. J. & Pearson, W. R. Rapid and sensitive protein similarity searches. Science 227, 1435–1441 (1985).</li>
	<li>Wang, L., Wang, S. & Li, W. RSeQC: quality control of RNA-seq experiments. Bioinformatics 28, 2184–5 (2012).</li>
	<li>Yoo, A. B., Jette, M. A. & Grondona, M. Slurm: Simple linux utility for resource management. in Workshop on Job Scheduling Strategies for Parallel Processing 44–60 (Springer, 2003).</li>
	<li>Li, H. et al. The sequence alignment/map format and SAMtools. Bioinformatics 25, 2078–2079 (2009).</li>
	<li>Stekhoven, D. J. & Buhlmann, P. MissForest--non-parametric missing value imputation for mixed-type data. Bioinformatics 28, 112–8 (2012).</li>
	<li>Massey Jr, F. J. The Kolmogorov-Smirnov test for goodness of fit. Journal of the American statistical Association 46, 68–78 (1951).</li>
	<li>Benesty, J., Chen, J., Huang, Y. & Cohen, I. Pearson correlation coefficient. in Noise reduction in speech processing 1–4 (Springer, 2009).</li>
	<li>Hauke, J. & Kossowski, T. Comparison of values of Pearson’s and Spearman’s correlation coefficients on the same sets of data. Quaestiones geographicae 30, 87–93 (2011).</li>
	<li>Kroger, C. et al. An infection-relevant transcriptomic compendium for Salmonella enterica Serovar Typhimurium. Cell Host Microbe 14, 683–95 (2013).</li>
	<li>Colgan, A. M. et al. The Impact of 18 Ancestral and Horizontally-Acquired Regulatory Proteins upon the Transcriptome and sRNA Landscape of Salmonella enterica serovar Typhimurium. PLoS Genet 12, e1006258 (2016).</li>
</ol> -->
