# Multi Layer Perceptron Classifier Analysis on UCI Datasets
- *Author:* Henrique de Morais Porto
- *Description:* This project implements and evaluates [Multi-Layer Perceptron classifiers](https://scikit-learn.org/1.5/modules/neural_networks_supervised.html#multi-layer-perceptron) on 3 datasets from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/). The focus is on datasets with categorical attributes and more than 3 classes. The project uses one-hot encoding for preprocessing and evaluates models using metrics like accuracy, precision, recall, and F1-score. The results are provided for each dataset, showcasing the MLP's performance in handling classification tasks.

## How to install and run? (Windows CMD)

```shell
cd .\"Trabalho Multi-Layer Perceptron"
```
```shell
python -m venv env
```
```shell
env\Scripts\activate
```
```shell
pip install -r requirements.txt
```
Replace ```{dataset_number}``` with the dataset folder number:
```shell
cd .\"Dataset {dataset_number}"
```
```shell
python main.py
```
## Dataset 1:
- Title: Car Evaluation
- Description: 
```
Derived from simple hierarchical decision model, this database may be useful for testing constructive induction and structure discovery methods.
```
- Link: https://archive.ics.uci.edu/dataset/19/car+evaluation
- Results:
```
Accuracy: 0.9942
Precision: 0.9943
Recall:    0.9942
F1 Score:  0.9942
Relatório de Classificação:
              precision    recall  f1-score   support

         acc       0.99      0.98      0.99       115
        good       0.95      1.00      0.98        21
       unacc       1.00      1.00      1.00       363
       vgood       0.95      0.95      0.95        20

    accuracy                           0.99       519
   macro avg       0.97      0.98      0.98       519
weighted avg       0.99      0.99      0.99       519
```

## Dataset 2:
- Title: Soybean (Large)
- Description: 
```
Michalski's famous soybean disease database. There are 19 classes, only the first 15 of which have been used in prior work. The folklore seems to be that the last four classes are unjustified by the data since they have so few examples. There are 35 categorical attributes, some nominal and some ordered.  The value "dna'' means does not apply.  The values for attributes are encoded numerically, with the first value encoded as "0,'' the second as "1,'' and so forth. An unknown values is encoded as "?''.
```
- Link: https://archive.ics.uci.edu/dataset/90/soybean+large
- Results:
```
Acurácia: 0.8750
Precisão: 0.8800
Recall:   0.8750
F1 Score: 0.8736
Relatório de Classificação:
                        precision    recall  f1-score   support

   alternarialeaf-spot       0.62      0.83      0.71        12
           anthracnose       1.00      1.00      1.00         6
      bacterial-blight       1.00      1.00      1.00         3
     bacterial-pustule       1.00      1.00      1.00         3
            brown-spot       1.00      0.92      0.96        12
        brown-stem-rot       1.00      1.00      1.00         6
          charcoal-rot       1.00      1.00      1.00         3
 diaporthe-stem-canker       1.00      1.00      1.00         3
          downy-mildew       1.00      1.00      1.00         3
    frog-eye-leaf-spot       0.70      0.58      0.64        12
phyllosticta-leaf-spot       0.50      0.33      0.40         3
      phytophthora-rot       1.00      1.00      1.00         5
        powdery-mildew       1.00      1.00      1.00         3
     purple-seed-stain       1.00      1.00      1.00         3
  rhizoctonia-root-rot       1.00      1.00      1.00         3

              accuracy                           0.88        80
             macro avg       0.92      0.91      0.91        80
          weighted avg       0.88      0.88      0.87        80
```
## Dataset 3:
- Title: Nursery
- Description: 
```
Nursery Database was derived from a hierarchical decision model originally developed to rank applications for nursery schools. It was used during several years in 1980's when there was excessive enrollment to these schools in Ljubljana, Slovenia, and the rejected applications frequently needed an objective explanation. The final decision depended on three subproblems: occupation of parents and child's nursery, family structure and financial standing, and social and health picture of the family. The model was developed within expert system shell for decision making DEX (M. Bohanec, V. Rajkovic: Expert system for decision making. Sistemica 1(1), pp. 145-157, 1990.).

The hierarchical model ranks nursery-school applications according to the following concept structure:

 NURSERY            Evaluation of applications for nursery schools
 . EMPLOY           Employment of parents and child's nursery
 . . parents        Parents' occupation
 . . has_nurs       Child's nursery
 . STRUCT_FINAN     Family structure and financial standings
 . . STRUCTURE      Family structure
 . . . form         Form of the family
 . . . children     Number of children
 . . housing        Housing conditions
 . . finance        Financial standing of the family
 . SOC_HEALTH       Social and health picture of the family
 . . social         Social conditions
 . . health         Health conditions

Input attributes are printed in lowercase. Besides the target concept (NURSERY) the model includes four intermediate concepts: EMPLOY, STRUCT_FINAN, STRUCTURE, SOC_HEALTH. Every concept is in the original model related to its lower level descendants by a set of examples (for these examples sets see http://www-ai.ijs.si/BlazZupan/nursery.html).

The Nursery Database contains examples with the structural information removed, i.e., directly relates NURSERY to the eight input attributes: parents, has_nurs, form, children, housing, finance, social, health.

Because of known underlying concept structure, this database may be particularly useful for testing constructive induction and structure discovery methods.
```
- Link: https://archive.ics.uci.edu/dataset/76/nursery
- Results:
```
Accuracy: 0.9997
Precision: 0.9995
Recall:    0.9997
F1 Score:  0.9996
Classification Report:
              precision    recall  f1-score   support

   not_recom       1.00      1.00      1.00      1296
    priority       1.00      1.00      1.00      1280
   recommend       0.00      0.00      0.00         1
  spec_prior       1.00      1.00      1.00      1213
  very_recom       0.99      1.00      0.99        98

    accuracy                           1.00      3888
   macro avg       0.80      0.80      0.80      3888
weighted avg       1.00      1.00      1.00      3888
```