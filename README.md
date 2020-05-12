# EE148Project

### ssh command line 
ssh -L 5901:localhost:5901 -i "ee148.pem" ubuntu@ec2-54-177-211-159.us-west-1.compute.amazonaws.com



### train_data_selection.ipynb

Selects artworks in train_1 dataset with valid date feature

Generates train_1_info.csv  (columns: ["filename", "date"])



### dataset.ipynb

Implements a pytorch dataset.

Resizes the pictures.

Converts PIL Images to pytorch tensors.

