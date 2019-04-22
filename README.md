# Image Text Labeling - Proof Of Concept #

This is a POC for a technique to label text in a set of images, such as invoices, particularly where the meaning of some text, such as values, 
can only be ascertained by reference to adjacent text tags.  E.g. labeling as "Total", the value "$100" in an invoice because of its position to the tag "Amount payable"
It takes the output from the Google Vision api, and applies some transformations and a machine learning model to classify the output.

Initial testing was on a handful of fields in a set of 1000 made up invoices, with 5 basic templates.  The amount of variation in the invoices was controlled to
present a moderate challenge.  With this limited data set split into training set of 800 and test set of 200 and a basic model, results were promising with 99% labeling accuracy achieved on the test set.

For second test run, the same handful of fields were used, but the training set was generated as tables using python, producing more variation over 10,000 invoices.
It still obtained 99% accuracy on the test set, and is promising in identifying those fields on a handful of images generated by a different mechanism

The training data generation at the moment is still fairly limited.  E.g. the values, "$100" are always positioned to the right of the tag "Amount payable"

### Overview ###

This code implements a machine learning solution to assign meaning labels to the words output by the Google Vision api. https://cloud.google.com/vision/

The Google Vision output is in JSON containing for each word detected the location on the document as x, y co-ordinates of the 4 corners of a boundary box.
This is easily converted to a wor-location table, Example:

text	x-top-left	y-top-left	x-top-right	y-top-right	x-bottom-right	y-bottom-right	x-bottom-left	y-bottom-left
Invoice	1183.0	144.0	1447.0	144.0	1447.0	199.0	1183.0	199.0
Amount	832.0	214.0	876.0	214.0	876.0	234.0	832.0	234.0
Payable	884.0	214.0	913.0	214.0	913.0	231.0	884.0	231.0
$100	922.0	214.0	960.0	214.0	960.0	231.0	922.0	231.0

The following transformations as then applied:-
	- The words are encoded as vectors using GloVe plus a few other flags, e.g. is_number
	- The locations are used to create a 2d grid recording where the words are located
	- The grid is expanded to a 3d array with one channel for each component of the word vectors.
		Pictorially the grid is pseudo colored using the vectors as pseudo primary colors
The data now looks mathematically like a standard image, except it has >50 channels from the word vectors instead of 3 primary color channels, so standard computer vision techniques can be applied
A basic CNN model is used to segment the pixels, predicting the labeling for each pixel.
The label prediction is combined with the grid, which acts as a word map, to match the pixel label predictions to the original text

Example code and tag lists are provided to generate basic training data as word-location text files plus label files.

The code was written on Windows with Spyder as the IDE within Anaconda and has not been tested in other environments.

A deeper explanation is available in the following slides https://docs.google.com/presentation/d/e/2PACX-1vTNhfucXtqEhNGh_TUh6r4I6DehdNkG1c4JOQ1PtvbU2DBKz8ydbuaHcqsM8bmdOYsYarYTR5RnVt7R/pub?start=false&loop=false&delayms=3000

### Prerequisites ###
	Python 3.x
		numpy
		pytorch (040)   https://pytorch.org/
		matplotlib
	
	b64   for converting images to base 64   https://sourceforge.net/projects/base64/
	curl  for submitting base 64 images to Google Vision  https://curl.haxx.se/
	a Google Cloud Platform apikey in a text file, pointed to in the local_config file  https://cloud.google.com/vision/docs/auth#using_an_api_key
	
	GloVe or equivalent word vectorization of the language.  glove.6B.50d.txt was adequate in initial testing  https://nlp.stanford.edu/projects/glove/
	
	A list of English words is useful to generate training data.  https://github.com/first20hours/google-10000-english  was used.

	Excel spreadsheets are provided for a manual labeler and as a map analysis tool, but these are not necessary to run the code.
	
### Set up ###

	Create a copy of local_config_default.txt called local_config.txt within the repo subfolder and amend it to record where the components are located on your system
	as they are set up below.  Text in square brackets is the Key in the local_config file against which the file name/path should be recorded

	Install Python 3.x and modules above
	Install B64 and curl.  These do not need an environment path as the code will call them by the full path [B64] [CURL]
	Create an apikey for authorization to Google Vision and save in a local file [KEYFILE]
	Save GloVe file to a local file [WORD_VEC]
	Save a list of common words to a local file for training data generation, e.g. from google-10000-english [WORDS]
	Set up local folders for:-
		Training data [TRAINING_DATA]
		Temp files for use in prediction [TEMP_DATA]
		Images to be predicted [IMAGES]
			Put the example invoice inv1.png in the example_data/images subfolder into this folder to test with this
		Tags to be used in training data generation [TRAINING_DATA_GENERATION]
			Populate this with a tags file and lists of tags to be used to generate the training data.  See the example_tags subfolder for examples 
		Models [MODEL]
	Create a master file for labels, see Example_Labels_Master.csv for an example and explanation below [LABEL_MASTER]
	
	Run config.py to make the subfolders needed for training data


### Getting Started ###

With the above set up, using the example tags files it should be possible to train a model by:-
	running training data generation (generate_training_data.py) one cell at a time in say spyder
	manually splitting the generated data into train and test sets by moving the data to the array_data train and test subfolders, and
	running ml.py one cell at a time to train the data, building a new model as there won't be one to load
It should then be possible to run the label prediction of the example invoice inv1.png from the command line with: python predict.py inv1.png -a -r -v

### Label Master file format ###

	A csv file with a header row and 2 columns, for ID and Labels.
	Leave IDs 0 and 1 as per sample below.  Others should follow sequentially as desired
	The number of rows in the table drives the Output size of the model
	
ID,Labels
0,Not_a_word
1,Unassigned
2,Total
3,Total_Label

### Training data and labels ###

	Training data for the model consists of numpy arrays, of transformed word-location tables and labels
	The file table_to_array.py generates the arrays from word-location table and label text files.
	
	Word-location tables and label text files can be generated by either:-
	1) Taking a sample of images, running through Google vision, using the data_preparation.py file and manually creating the labels, e.g. with the manual_labeller.xlsm spreadsheet, or
	2) Generating the text files directly via python, based on lists of tags, random values and a word list to sample to add random unassigned blocks to complete the documents.
	The second method has the potential to generate a suitable large training data set.  Example code is provided in generate_training_data.py and tags in the example_tags subfolder
	

	Example formats
	The word-location table output is formatted:-
text	x-top-left	y-top-left	x-top-right	y-top-right	x-bottom-right	y-bottom-right	x-bottom-left	y-bottom-left
Document	1183.0	144.0	1447.0	144.0	1447.0	199.0	1183.0	199.0
Amount	832.0	214.0	876.0	214.0	876.0	234.0	832.0	234.0
Due	884.0	214.0	913.0	214.0	913.0	231.0	884.0	231.0
10,000	922.0	214.0	960.0	214.0	960.0	231.0	922.0	231.0

	Labels need to be a headed csv file like this, corresponding to the word-location table
Labels
Unassigned
Total_Label
Total_Label
Total


### Generating training data - 1 - from raw images ###

	Create a training folder, and record its location in the local_config.txt file under TRAINING_DATA
	Run the config.py file to generate suitable subfolders

	Place images to be used in the images sub folder of the training data folder
	
	from data_preparation.py   Prepare the data for training by running the following 
	    image_to_base64.convert_all_files    to convert image files to base 64 to be in a suitable format for submitting, saved to the imagesbase64 subfolder
	    base64_to_json.read_all_images       submits base64 images to Google Vision api and gets json in return saved to the json subfolder
	    json_to_table.create_all_tables      converts the json to a table, into the tabledata subfolder
		
	Put appropriate csv labels in the labels subfolder.  
		If the image name is img1.png then the corresponding label should be named img1_labels.csv
		An Excel file manual_labeller.xlsm is provided to facilitate creation of the appropriately formatted label files manually from the table files.

	from data_preparation.py    Create arrays by running:
	    table_to_array.create_all_arrays    converts the table and the labels if they exist to numpy arrays in a format suitable for machine learning, encoding words with GloVe
			This creates 3 array sets:-
				data arrays in the array_data subfolder     3d numpy arrays encoding the image and location with GloVe encoding plus enhancements for each word along the channel dimension
				label arrays in the array_labels subfolder   labels correspond to the indices from the tag master csv file.  0 corresponding to no word
				word location map in the array_wordmap subfolder  -1 means no word maps to that pixel

### Generating training data - 2 - by python ###

	Create a training folder, and record its location in the local_config.txt file under TRAINING_DATA
	Run the config.py file to generate suitable subfolders

	Example code to generate some training data is supplied in the generate_training_data.py file
	Run each cell in turn to generate the data.
	To use this file as is, create a tags folder, populate it with lists of tags as per the example_tags subfolder and record its location in the local_config.txt file under TRAINING_DATA_GENERATION

	The last step in the process is table_to_array.create_all_arrays as per the previous section

### Training the model ###

	Generate suitable training data as above

	Manually split the data arrays from the array_data subfolder to the train and test subsubfolders to create train and test sets
	
	from ml.py	Run the code blocks in order to train the data.  This uses a keras style syntax.  The code was written using spyder and the block syntax reflects this.
	
	An example file from each stage can be seen in the example_data subfolder of the repo

### Evaluating an image ###

	An image in the training data image subfolder can be evaluated from from the command line run with the following:
	python predict.py imagename -a
	
	This calls code in main.py which controls the refresh, and displays the response on screen
	imagename can be the full path and file name.  If the path is ommitted it looks in the images subfolder.  File extension defaults to .png if not specified
	
	Command line options
		-a       to print form each label, excluding unassigned, the text associated with that label
		-r       to print the full response, showing the label assigned to each word and the confidence in the prediction as a probability
		-s       to save the response to the response training directory
		-p path  to save to the specified path.  if image is inv1.png then response file name is inv1_response.txt
		-v       verbose to show progress, though the whole process isn't really long enough to need it.  Most of the time is in loading python.
	
	Alternatively from python call main.predict_an_invoice(filename)
		Additional optional parameters: verbose = True, save = False, path = config.pathresponse, print_response = False)

### Other functionality and files ###
	
	map_analyzer.xlsm is available to load the word maps to be able to visualize how the words are mapped to the grid
		