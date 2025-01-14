# Leslie-UnderGraduateProject-Protein-Classifier

This repository presented my undergraduate proect of classifying protens into one of 15 classes using machine learning models. The project involves steps of Read Protein Dataset, Preprocess Data, Handle Imbalanced Data, Apply Machine Learning Classifers, Stacking.

## Steps In Detail

### Read Protein Dataset

- Import the dataset with pd.read_csv
- Set the last column to be the label column, columns before to be the features columns
- The label column ranges from 0-14, and there are 2961 features columns before preprocessing
- Split the dataset to 25% test data, 75% train data

### Preprocess Data

- Drop all columns with only NaNs
- Fill NaN with medium for numeric columns
- Fill NaN with the most frequent value for categorical columns
- Encode yes/no to 1/0 for columns(Class, Complex, Phenotype, Motif)
- Apply one-hot encoding for columns(Essential, Interaction, Chromosome)
- Drop columns which contain only one distinct value
  
Now the dataset only consist with 1/0, by printing the first few columns, the dataset looks like
<img src="Llama-3.2-3B-Instruct training_loss.png" alt="Llama" style="width: 60%; min-width: 300px; display: block; margin: auto;">
