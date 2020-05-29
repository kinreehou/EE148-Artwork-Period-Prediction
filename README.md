# EE148Project

### ssh command line 
ec2-3-101-79-78.us-west-1.compute.amazonaws.com ssh -i "ee148.pem" ec2-user@ec2-3-101-79-78.us-west-1.compute.amazonaws.com



### train_data_selection.ipynb

Selects artworks in train_1 dataset with valid date feature

Generates train_1_info.csv  (columns: ["filename", "date"])



### dataset.ipynb

Implements a pytorch dataset.

Resizes the pictures.

Converts PIL Images to pytorch tensors.

