To train our model, the following command should be ran:
python training_script.py --dataset dataset --model "model.model"
Behind --dataset you should input the dataset for training and validation
Behind --model you should name the output model.
		output model will be saved under the same directory.


To test our model, the following command should be ran:
python test_masks.py --images dataset --model "hh.model"

Behind --dataset you can give the directory of the dataset 
and behind --model you can name the model and the model file
can be saved into the same directory as your script

---------------------------------------------------------------

To test one image of our model, the following command should be ran:
python test_mask.py --image image1.png --model "hh.model"

Behind --image input the image you want to test on
Behind --model you can name the model and the model file
can be saved into the same directory as your script