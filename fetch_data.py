# Main libraries
import requests
import pandas as pd
import numpy as np
import requests
import json
import pprint

# Conficg file
from configs import *


#### ------------------------------------------------------------------------------------------------------- ####


########################### Start The main functions of fetching and handle the data ######################################


########################### Start to convert the datatype of some column

def convert_column_to_string(column_to_convert):
    """
    The function used to convert all values of some column to string.
    
    Argument
        column_to_convert : pandas series, Which column you need to apply the conversion on.
    
    Return
        column_converted   : pandas series, The same column but now its has a values data type of string instead of integer
    """

    try:
        column_converted    = column_to_convert.astype(str)
        # These prints help me knows there is no missing in data while we apply these transformation
        print("We have total number of ids: ", len(column_converted))
        print("="*50)
    
    # In case of error sent to main logs direction
    except Exception as e:
        file                = open("logs/fetch_data.log","+a")
        file.write("This error related to function convert_column_to_string of fetch_data file \n"
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines
        
    return column_converted

########################### End of convert the datatype of some column


########################### Start to send the request with list of ids as strings

def request_ids(list_of_ids, url=URL):
    """
    The function used to request list of ids as json POST request to some url,
    and response with content related to these ids.

    Argument
        list_of_ids   : list, the list of ids you need to retrieve the corresponding text for.
        url           : string, The api url to call for retrieve the data

    Return 
        response_text : dictionary, json dictionary of the tweets associated with its ids

    """

    try:

        # Request some ids 
        response        = requests.post(url, json=list_of_ids)


        response_text = response.content

        # Convert the response text from string to json dictionary
        response_text = json.loads(response_text)

    # In case of error sent to main logs direction
    except Exception as e:
        file                = open("logs/fetch_data.log","+a")
        file.write("This error related to function request_ids of fetch_data file \n"
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines

    return response_text
    

########################### End of The main functions of fetching and handle the data ######################################


#### ------------------------------------------------------------------------------------------------------- ####



########################### Start the pipeline to collect the data from APIs ######################################

def fetching_pipeline(file_name_to_read, file_name_to_save, col_to_convert, dialect_col, data_path=DATA_PATH):
    '''
    The function used to combine all of our function into one pipeline that handle all of fetching data task.

    Arguments
        file_name_to_read : string, the name of the csv file you need to read from.
        file_name_to_save : string, the name of the csv file you need to save the new data after fetching it.
        col_to_convert    : string, Which column you need to convert its datatype to call the api.
        dialect_col       : string, The column of dialect to get it into the new file with new column.
        data_path         : string, The main dierction of our data.

    Return
        True              : boolean, in case there is no error occured in our pipeline.
    '''
    try:

        # Reading the dialect_dataset using read_csv function defined in that file
        dialect_dataset  = read_csv(file_name_to_read)
        
        # Convert the value data types of id column into string using convert_column_to_string function defined in that file
        dialect_ids      = convert_column_to_string(dialect_dataset[col_to_convert])
        
        # Convert the returned dialect ids into list as it required by the APIs to make success request
        dialect_ids_list = list(dialect_ids)

        dialect_col      = list(dialect_dataset[dialect_col])

        # Start from first id with index 0
        start = 0

        # Collect the retrived text from API into one list
        all_retrieved_text_list = []

        for end in range(1000, len(dialect_ids_list), 1000):

            # As we have to call the API with Max length of list 1000, 
            # so we need to start from 0 to 1000, then from 1000 to 2000 and so on

            subset_of_ids = dialect_ids_list[start:end]
            start = end

            # Get the corresponding text for these subset of ids
            response_text  = request_ids(subset_of_ids)

            # convert just the values(text) without ids of response_text dictionary into list
            list_of_text = list(response_text.values())

            # Append to our main list that collect the text for all ids
            all_retrieved_text_list += list_of_text

            # Get the rest of ids in our list
            if len(dialect_ids_list) - end < 1000:
                subset_of_ids = dialect_ids_list[end:] # from the end we have to the end of the list
                response_text = request_ids(subset_of_ids)
                print("The type of the converted content of the ids are: ", type(response_text))
                print("="*50)
                list_of_text = list(response_text.values())
                all_retrieved_text_list += list_of_text

                # At the end display some of the result from the json dictionary
                n_items = display_json_result(response_text.items(), 10)
                pprint.pprint(n_items)

        # Create new dataframe with the retrieve text column as well as with other columns
        dialect_data_frame            = pd.DataFrame({"id": dialect_ids_list, "dialect":  dialect_col, "text": all_retrieved_text_list})
        # Save as new csv file to start the preprocessing pipeline on
        file_path_to_save = data_path + file_name_to_save
        dialect_data_frame.to_csv(file_path_to_save, index=False, encoding='utf8')
        
        print("Our fetching pipeline is work without any error.")

    except Exception as e:
        file                = open("logs/fetch_data.log","+a")
        file.write("This error related to function fetching_pipeline of fetch_data file \n"
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines
    return True


########################### End the pipeline to collect the data from APIs ######################################
