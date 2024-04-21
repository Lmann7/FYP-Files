The input sets referenced in the report and their corresponding file names:\
Spectral set = Mata-P3D-4-imputed.csv\
Spectral Input Set with Additional Variables = Mata-P3D-3-imputed.csv\
Initial Spectral+Derived = Mata-P3D-2-imputed.csv\
Improved Spectral+Derived = Mata-P3D-2.6-imputed

The model names have the form 'MataP3D_Model_2.6_E-S' or 'MataP3D_Model_2.6'\
The 2.6 part indicates the input file used, \
The part after that indicates the type of classification.\
E-S = Binary, elliptical vs. spiral\
Sab-Scd = Binary spiral model, Sa-Sb vs. Sc-Sd\
S2 = Multi-class spiral, 7 spiral sub-classifications\
Blank, e.g: 'MataP3D_Model_2.6' = Multi-class, elliptical vs. spiral vs. lenticular

The input file used for each model will be read in at the top of the script so if unsure \
just check the script and download the correct input file from FYP-Files/All Input Files

The code for the ionisation type models are in: FYP-Files/Ionisation Type Models
The code for the morphological type models are in: FYP-Files/Main Morphology Models
Additional models created are in: FYP-Files/Extra Models
Scripts for data processing and formatting are in: FYP-Files/Data processing-formatting
Scripts for the analysis of the results from the models are in: FYP-Files/Analysis  
Most functions from the scripts contained in FYP-Files/Analysis are already implemented into the model creation scripts 
