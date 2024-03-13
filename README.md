# Generalized Propensity Learning

## Experiments on real-world datasets

### Prerocess datasets
1. Download Coat from https://www.cs.cornell.edu/~schnabts/mnar
2. Run preprocessor.py

### Training Process

#### MRDR
###### Baseline
1. Run propensity.py to find the best l2_reg_lambda for CTR prediction and predict propensity scores.
2. Run MRDR-DL.py for CVR prediction and find the best l2_reg_lambda for CVR prediction:
   - Cross-Entropy, DCG and Recall are recorded in '../excel/mrdr/(dataset)/baseline/(baseline_mrdr_%s_%d_%d.xlsx)'
   - Find the file that records the lowest average cross-entropy and its corresponding l2_reg_lambda is the best one.

###### MRDR-GPL
1. Set the l2_reg_lambda for CVR prediction as the value of the best l2_reg_lambda for CVR found in the baseline experiment.
2. Run MRDR-DL-GPL.py

## Correction
03/13/2024

There is a typo in Eq.(28) of the paper. It should be divided by |D|^2 rather than |D|, as indicated in the last line of Eq.(27). The correct equation can be found on page 17 of the slides.
