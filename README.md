# EBCG : Event-Based Commentary Generator

This is a working repository with things updating frequently. The nature of the task is experimental so there is no set structure for now. 

## Dependencies

> Install dependencies either using requirements.txt or environment.yml
> 
> To use pip, run : pip install -r requirements.txt
> 
> To use conda, run :
>> conda env create -f environment.yml
>> 
>> conda activate ebcg
## Data Collection

> To view the data collection phase, look at **extract.ipnyb**
> 
> The file, **match_pages.txt** contains list of the url's used by extract.ipnyb to retrieve event and commentary data.
> 
> The file, **event_data.pickle** contains the event and the commentary data.

## Model Experiments

> To experiment with the GPT-2 model, please visit [here](https://drive.google.com/drive/folders/1v6E_4W_cbpgS0_CvqAE0jaksoYJSkflc?usp=sharing)
> 
> The file, **model.ipnyb**, loads the data and quickly trains fastai's tabularlearner for entity categorical embedding and can also predict and output commentary. It also contains code for the implementation of [Taniguchi et al](https://ojs.aaai.org//index.php/AAAI/article/view/4691)
> 
> The file, **model1.ipnyb** can be ignored for now as it is further attempt to implement the above paper.
> 
> The file, **bi_lstm_model.ipnyb** implements a class for Bi-Directional LSTM used for generating commentaries. It should not be executed right now as it uses the whole of sequence data from commentaries for training and testing with a total size of 5,069,635 with window set to 100.
> 
> The file, **process_data.py** implements a class to process the data into tokens and build the word2index and index2word dictionaries and to create sequences and targets.
>
> The file, **test_Bi_LSTM.ipnyb** is testing the implementation of the Bi-Directional LSTM model.
