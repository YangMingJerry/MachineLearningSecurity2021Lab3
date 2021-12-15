# MachineLearningSecurity2021Lab3
This is code for lab3

Ming Yang
my2153

## 1. How to get repaired models?
1. check the models saved at /models
2. Or you can also Run the pruning.py script to get the 2,4,10 acc dropping models.
3. The accuracy and asr curve is /prune_history.png
## 2. How to evaluate the goodnet?

Run evaluation code for good net from terminal: 

To check result for the first image of the backdoored dataset:
```shell
python goodnet.py
```
To check any image, put the path to the image file after that line:

```shell
python goodnet.py data/test_img/bd_0.jpeg
```
The output would be like:

the predicted label of this image is [1284]
