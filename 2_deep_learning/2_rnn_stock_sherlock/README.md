# Recurrent Neural Networks projects: time series prediction and text generation


## Run
In a terminal or command window run the following command:

  jupyter notebook RNN_project.ipynb


## Amazon Web Services

Instead of training your model on a local CPU (or GPU),  used Amazon Web Services to launch an EC2 GPU instance.  Please refer to the Udacity instructions for setting up a GPU instance for this project.


## Rubric items

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Submission Files      |  RNN_project.ipynb, my_answers.py --> both the completed notebook  RNN_project.ipynb as well as all completed python functions requested in the main notebook RNN_project.ipynb (TODO items) should be copied into this python script and submitted for grading.		|

#### Step 1:  Implement a function to window time series
| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Window time series data. |  The submission returns the proper windowed version of input time series of proper dimension listed in the notebook.  |


#### Step 2: Create a simple RNN model using keras to perform regression

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Build an RNN model to perform regression. |  The submission constructs an RNN model in keras with LSTM module of dimension defined in the notebook.        |


#### Step 3: Clean up a large text corpus

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Find and remove all non-english or punctuation characters from input text data.  The submission removes all non-english / non-punctuation characters.  |


#### Step 4: Implement a function to window a large text corpus

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Implement a function to window input text data| The submission returns the proper windowed version of input text of proper dimension listed in the notebook.  |


#### Step 5: Create a simple RNN model using keras to perform multiclass classification

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Build an RNN model to perform multiclass classification. |  The submission constructs an RNN model in keras with LSTM module of dimension defined in the notebook.        |


#### Step 6: Generate text using a fully trained RNN model and a variety of input sequences
| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Generate text using a trained RNN classifier.   | The submission presents examples of generated text from a trained RNN module.  The majority of this generated text should consist of real english words. |
