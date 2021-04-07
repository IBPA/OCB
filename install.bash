#!/bin/bash
#
# The script that install all required toolkits for Omics-Compendium-Builder-
# OCB.
# 
# The toolkits will be downloaded and installed into user-defined directory, 
# and then add this directory to PATH environment variable so that the
# pipeline can use it.
#
# If you want to install toolkits via apt with sudo privileges, please
# install those toolkits manually.
# 
# History:
# 2021/04/04 Tan, ChengEn (cetan@ucdavis.edu)

# The toolkits will be downloaded and installed here.
# Change the path if necessary
install_path="$HOME/Omics-Compendium-Builder-OCB"

mkdir -p $install_path
cd $install_path

# Check python installation

# If your Ubuntu use python2 as the default python version, uncomment these
# two lines so that Ubuntu can use python3 as the default python version
# (change the python3 you are using if necessary)
#shopt -s expand_aliases
#alias python="/usr/bin/python3"

python --version &>/dev/null

if [ ! $? -eq 0 ]
then
	echo "Please install python3 first." 
	exit -1
fi

main_version=$(python --version 2>&1 | cut -d " " -f2 | cut -d "." -f1)

if [ "$main_version" -ne 3 ]
then
	echo "Please use python3."
	exit -1
fi

sub_version=$(python --version } cut -d " " -f2 | cut -d "." -f2)
if [ "$sub_version" -lt 6 ]
then
	echo "Please use python 3.6 or later."
	exit -1
fi

#Test pip module
python -m pip --version > /dev/null
if [ $? -eq 0 ]
then
	echo "Pip is ready."
else
	echo "Pip is not ready -- please install pip first!"
	exit -1
fi

# 1. Download and unzip sratoolkits
wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.10.8/sratoolkit.2.10.8-ubuntu64.tar.gz
if [ $? -eq 0 ]
then
	echo "SRA toolkit is downloaded successfully."
else
	echo "SRA toolkit download failed." 
	echo "Please inform the author if it occurred constantly."
	exit -1
fi

tar -xvf sratoolkit.2.10.8-ubuntu64.tar.gz

# 1.1 Configure sratoolkits so that user can download the sra file directly
#     without additional configuration
#     NOTE: the following line is a trick so that the vdb-config can be run automatically
echo "" | sratoolkit.2.10.8-ubuntu64/bin/vdb-config -i | tee -a /dev/null
#     Reset the terminal (Which is necessary otherwise the terminal cannot print next messages)
tput reset

# 1.2 Test the configuration is complete
echo "Now try to run the prefetch command..."

sratoolkit.2.10.8-ubuntu64/bin/prefetch --version
if [ $? -eq 0 ]
then
	echo "SRA configure OK!"
else
	echo "SRA configure failed. Please inform the author"
	exit -1
fi

# 2. Download and install bowtie2
wget https://sourceforge.net/projects/bowtie-bio/files/bowtie2/2.3.4/bowtie2-2.3.4-linux-x86_64.zip/download -O bowtie2-2.3.4-linux-x86_64.zip

if [ $? -eq 0 ]
then
	echo "Bowtie2.3.4 is downloaded successfully."
else
	echo "Bowtie2.3.4 download failed." 
	echo "Please inform the author if it occurred constantly."
	exit -1
fi

unzip -o bowtie2-2.3.4-linux-x86_64.zip
if [ ! $? -eq 0 ]
then
	exit -1
fi

# 2.1 Test the bowtie2 installation
bowtie2-2.3.4-linux-x86_64/bowtie2 --version > /dev/null
if [ $? -eq 0 ]
then
	echo "Bowtie2 installed successfully"
else
	echo "Bowtie2 installed failed. Please inform the author."
fi


# 3. Install necessary python packages
python -m pip install RSeQC==3.0.0 --user
if [ ! $? -eq 0 ]
then
	echo "Failed to install RSeQC 3.0.0."
	echo "If <zlib.h> not found, Please try to install zlib library and try again."
	echo "To install zlib, use the following command:"
	echo "sudo apt-get install zlib1g-dev"
fi

# Exit if one installation failed
set -e
python -m pip install biopython==1.74 --user
python -m pip install pandas==0.25 --user
python -m pip install HTSeq==0.11.2 --user
python -m pip install missingpy==0.2.0 --user
python -m pip install scikit-learn==0.20.1 --user
python -m pip install matplotlib==3.0.2 --user


echo "Python packages installed successfully!"
echo "All required packages and toolkits are installed successfully!"
echo ""
echo "Please add these two lines to ~/.bashrc to set the environment variable:"
echo ""
echo "    export PATH=$install_path/sratoolkit.2.10.8-ubuntu64/bin/:\$PATH"
echo "    export PATH=$install_path/bowtie2-2.3.4-linux-x86_64/:\$PATH"
echo ""
echo "Then, restart the terminal, or run the following command to update the environment variable:"
echo ""
echo "    source ~/.bashrc   "
echo ""
