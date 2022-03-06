from itertools import islice
import pandas as pd

# Main directions
DATA_PATH = "dataset/"
URL = 'https://recruitment.aimtechnologies.co/ai-tasks'

#### ------------------------------------------------------------------------------------------------------- ####


########################### Start to read csv file

def read_csv(file_name, data_path=DATA_PATH):
    '''
    The function used to read csv file.
    
    Argument
        file_name       : string,   The path of the file we need to reed.
    Return
        dialect_dataset : datafame, The readed file as dataframe.

    '''
    try:
        dialect_dataset = pd.read_csv(DATA_PATH + file_name, lineterminator='\n')
        print("Number of instances in the file are: ", len(dialect_dataset))

    except Exception as e:
        print("You need to first handle the error related to reading the data to keep gooing: \n", e)
        

    return dialect_dataset

########################### End of read csv file


########################### Start to display some items from dictionary

def display_json_result(iterable, n):
    """
    The function used to display the respose from the APIs for some ids.

    Argument
        iterable : iterator, over the dictionary items.
        n        : int, how many items you need to display.
    Return
        n_items  : dictionary of n items to display the key is ID, and value is the text.
    """

    n_items = dict(islice(iterable, n))
    return n_items

########################### End of display some items from dictionary

########################### Start to validate the data used and the new created data with new text column

def validate_ids_and_dialect(dialect_dataset, new_dialect_dataset):
    '''
    The function used to ensure that we have not missed or change in the ids as well as the dialect between 
    the new created dialect_dataset with text and the data we used to call the APIs.

    Argument
        dialect_dataset      : The original dataset
        new_dialect_dataset  : The new created dataset
    Return
        True                 : boolean if there is no error occurred
    '''
    print("="*50)
    print("The columns of orginal data are: ", dialect_dataset.columns)
    print("="*50)
    print("The columns of new created data are: ", new_dialect_dataset.columns)
    print("="*50)

    # Retrieve columns data as list
    dataset_ids         = list(dialect_dataset['id'])
    dataset_dialect     = list(dialect_dataset['dialect'])
    new_dataset_ids     = list(new_dialect_dataset['id'])
    new_dataset_dialect = list(new_dialect_dataset['dialect'])
    
    for i in range(len(dataset_ids)):
        assert (dataset_ids[i]     == new_dataset_ids[i])
        assert (dataset_dialect[i] == new_dataset_dialect[i])


    return True

########################### End of validate the data used and the new created data with new text column



