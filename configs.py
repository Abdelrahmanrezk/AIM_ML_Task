from itertools import islice
import pandas as pd
from collections import Counter

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


def get_keys_that_val_gr_than_num(num_of_words_in_each_text, num):
    '''
    The function used to get dictionary that value of its keys are greater than some number.

    Argument
        num_of_words_in_each_text : list, The list to get the values repeated in as keys and how many times its repeated as value.
        num                       : int, Which keys its value grater than that num to save in your dictionary.
    Return
        new_dicts                 : dictionary, keys and its related repeated value greater than some num
    '''
    # get number of times the text has same number of tokens
    dicts = dict(Counter(num_of_words_in_each_text))

    # Get new object instead of reference to same dictionary as we do not need to delete of what we loop over.
    new_dicts = dicts.copy()

    print("The number of keys before removing are: ", len(new_dicts))
    print("="*50)
    for key, val in dicts.items():
        if val <= num:
            new_dicts.pop(key)

    print("The number of keys after removing some of them are: ", len(new_dicts))
    print("="*50)
    new_dicts = {key: val for key, val in sorted(new_dicts.items(), key=lambda item: item[1])}
    return new_dicts
    


########################### End of validate the data used and the new created data with new text column



def save_train_test_data(data, sub_dir, file_name_to_save, data_path=DATA_PATH):
    '''
    The function used to save the data after the spliting we apply to it.

    Argument
        data              : dataframe, the data you need to save.
        train_dir         : string, in which direction inside the main dataset direction you need to save your data.
        file_name_to_save : string, the csv file name you need to save the file with.
    '''
    # Get the full path to save the file

    file_path_to_save = data_path + sub_dir + file_name_to_save
    data.to_csv(file_path_to_save, index=False, encoding='utf8')

    return True