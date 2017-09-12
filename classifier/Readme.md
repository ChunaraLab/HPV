
The code is accompanied with template_marked_tweets.csv which provides the training/ labelled datasets for training. The newtag column contains the labels and can taking any integer value (not just a binary classifier)

The code is accompanied with template_unmarked_tweets.csv which provides a template of how the input of unmarked data should look like. The current file does have values in the newtag column, but it can be left empty as these labels are not used in the training or testing in anyway.

The code outputs a file template_predicted.csv which contains the labels for each tweet in the template_unmarked_tweets dataset.

