## DeepSense Model for Vehicle Recognition -- Version 1
### Replace the "train_data" folder with the one in Dropbox
Copy the files in Dropbox folder "ShakeData/GQ-2022-01-06-clean/train_data/" to "train_data/" folder in this repo:
train_X_both.csv
train_Y.csv  
val_X_both.csv
val_Y.csv
test_X_both.csv  
test_Y.csv             

### Environment
Python 3.7.10

Run "pip install -r requirements.txt" to install the required packages.

### Run the code
We save the model we trained in './log/weight_deepSense.h5', to load the model from './log/weight_deepSense.h5' and evaluate its performance, just run

"python eval.py -m log/weight_deepSense.h5"

If you want to train the model by youself, you can run

"python train.py -m 'model_path'"

'model_path' is the file to store the trained model with a default value './log/weight_default.h5'
