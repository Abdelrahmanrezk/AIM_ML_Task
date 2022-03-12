
# Main libraries 
from tashaphyne.stemming import ArabicLightStemmer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.preprocessing import LabelEncoder
from nltk import ngrams
import emojis
import re

# Conficg file
from configs import *




############## Start heuristic design of Regular Expression compiles to speed the process #############

# get urls in text 
URL_REPLACE        = re.compile(r"http\S+")

# get mentions in text
MENTIONED_REPLACE  = re.compile(r"@[A-Za-z0-9_]+")

# get more than one space between words
MORE_SPACES        = re.compile(r'([\s\t\n]+)')

# get chars repeated more than two times sequntially
CHAR_REPEATED      = re.compile(r'(.)\1+')

# remove dicrstics in text like(ً ُ)
TASHKEEL           = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

# make a space between numbers asscoiated with words 
SEP_NUM_WORD       = re.compile(r'(\d+)')

# make a space between English asscoiated with Arabic words 
SEP_EN_AR       = re.compile(r'[a-zA-Z]+')

# Uncide and emojis 
DETECT_SPECIAL_CHARS = re.compile("[\U0001f9bd\U0001f9d1\U0001f7e0\U000fe19b\U000fe19c\U000fe1b5\U000fe19d\U000fe19f\U000e0067\U0001f9b1\U000e007f\U000e0062\U000e006e\U000e0065\ue6fd\ue67b\ue67a\ue422\ue41f\ue327\ue224\ue40f\U000feb9d\ue703\U000fe527\ue677\ue222\ue221\uf8ff\ue694\U0001f7e1\U0001f90f\ue438\U000feb18\U0001f9dc\ue050\ue219\ue670\ue110\U0001f92e\U000feb5e\ue41d\ue32c\ue139\ue312\ue63e\ue748\U0001f7e2\ue046\ue033\ue112\U000fec17\ue335\ue035\ue44b\U000feb0d\U000fe32d\U000fe040\U0001f9e3\U000fe043\ue301\U0001fac2\ue009\U000fe354\ue00a\U000fe326\U000fe326\U000fe7f0\ue21d\ue21e\ue21f\ue444\ue66a\ue307\ue331\ue437\U0001f9a0\ue68d\ue405\ue21c\ue326\ue40c\ue03e\ue326\ue03e\U0001f9d6\ue304\U000fe987\ue681\U0001f993\ue305\ue427\ue303\ue118\ue404\ue32a\ue6ee\ue72d\ue6a4\ue22c\ue68d\U000fe1a8\ue20c\U0001fa84\ue04e\U000fe35b\U0001f9ed\U000fe54f\U000feb16\U000feb15\U000fe33d\ue704\ue105\ue01d\ue510\U000feb69\U000feb76\U000feb69\U000feb69\ue03f\ue12f\U000fe324\ue201\ue115\ue316\ue122\ue313\ue129\ue10f\ue03d\ue04a\ue02f\U000fe1b6\U000feb5c\ue058\ue71c\uf707\U000fe046\ue413\ue6e0\ue406\ue03c\ue042\ue411\ue410\U000fe333\U000fe041\ue402\ue6ec\ue72e\ue011\U000fe329\U000fe338\ue701\U000fe1b2\ue419\ue407\U000fe01f\U000feb75\U000feb85\ue022\U0001f9ba\ue347\ue32b\U000fe7d9\U000fe502\U0001f992\ue32e\U000feb5b\ue401\U000fe32a\U000fe1b3\U000fe341\U0001f9ce\ue314\ue725\ue32b\ue204\ue40e\ue00e\ue223\ue420\U0001f9d5\U0001f9e2\U0001fab0\U0001fac0\U0001f9e8\U0001f9d8\ue6ff\ue6fc\ue408\U0001f976\ue416\ue709\ue447\ue119\ue41b\U000fe351\ue418\ue148\U000fe335\ue030\ue31d\U0001fa78\ue747\ue702\ue306\ue106\ue698\ue057\ue529\U0001f91f\ue414\U000fe04d\U0001f9b9\U0001f90c\uf60d\U000fe532\ue67f\U000fe4f5\U0001f9e4\ue027\U000fe34a\ue409\ue22f\ue40a\ue049\ue04b\U000fe347\ue00f\U0001f9d4\ue754\ue311\ue113\U0001f964\ue11c\U000feb99\U000fec00\ue700\U0001f9e6\U000fe526\ue432\ue695\ue108\ue13c\ue319\ue676\U000fe03d\ue012\ue20f\ue220\U0001f972\ue6ac\U0001f966\ue643\u2060\U0001f9cd\U0001f9a7\U0001f962\ue328\ue71b\U000feb0f\U000feb6a\ue107\U000fe344\uf815\ue421\ue684\U0001f92f\U0001f9a6\U000fe358\ue6f6\ue008\U000feb9a\ue01b\U000fe340\ue70a\ue41e\U000feb55\ue056\ue32d\ue328\ue40d\ue337\ue743\ue417\ue023\ue403\ue6ef\ue6f0\ue415\ue056\ue022\ue412\U0001f92b\U000feb14\U0001f9b6\U000fe339\U000fe320\U000fe323\U000feb0e\U000fe342\U0001f92c\U000fe512\U000fe517\U000fe520\U000fe331\U000fe32c\U0001f971\U0001f9d0\U000fe33a\U0001f9ff\U0001f94d\U0001f975\U000fe19e\U000fe541\U000feb13\U0001f929\U000fe32f\U000fe343\U0001f9da\U0001f9af\ue757\U000fe1ab\U000fe1a5\U0001f928\U0001f9a5\U0001f974\U0001f973\U0001f90e\U0001f92a\U000feb9f\U000feb97\U000fe330\U000feb9e\U000fe334\U000feb96\ue537\U000fe327\U000fe334\U000fe33e\u200c\u200d\u202c\u200e\u202b\U0001f90d\u2067\u200f\u200b\U0001f97a\U0001f932\u2066\u2069\U0001f90d\U0001f970\U0001f92d\U0001f9e1\ue032\ue059\u06dd]")

# ############## End heuristic design of Regular Expression compiles to speed the process #############

#### ------------------------------------------------------------------------------------------------------- ####




########################## Start to Clean / Normalize Arabic Tweets

def clean_text(text):
    '''
    The function used to Clean / Normalize Arabic Text:
        - Replace url with Non-standard Arabic name
        - Replace mentions with Non-standard Arabic name
        - Remove TASHKEEL (Special chars for Arabic language)
        - Remove part of appeared special chars like /U000f and others in DETECT_SPECIAL_CHARS defined in the begining
        - Some special replacement and handle the other replacement in the two list [search & replace]
        - Multiple emojis coming sequentially leave just one of them
        - Separate numbers associated with words as well as English with Arabic words
        - Remove char repeated more that two times sequentially
        - Remove more spaces
        
        
    Argument
        text: string, The text we need to handle
    Return
        text: string, the cleaned tweet after preprocessing it.
    '''
 
    search  = ['&quot;', "أ", "إ", "آ", "ة", "/", "!", '"', "'", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", ":", ";", "<", "=", ">", "?", "؟", "[", "]", "^", "_", "`", "{", "}", "|", "~", "ى","\\",'\n', '\t']
    replace = [' ', "ا", "ا", "ا", "ه", " / ", " ! ", "", "", " $ ", "%", " & ", " ( ", " ) ", " * ", " + ", " , ", "-", " . ", " : ", " ; ", " < ", " = ", " > ",  " ? ", " ؟ ", " [ ", " ] ", " ^ ", "_", " ` ", " { ", " } ", " | ", " ~ ", "ي"," \\ ",' ', ' ']
    
    # Replace the url with non-standard Arabic word 
    text = URL_REPLACE.sub(r"رابطويب", text)

    # Replace the mention with non-standard Arabic word 
    text = MENTIONED_REPLACE.sub(r"حسابشخصي", text)

    # remove tashkeel
    text = TASHKEEL.sub(r"", text)
    

    # By practice remove most appeared special chars like \U000f
    text = DETECT_SPECIAL_CHARS.sub('', text)
    
    # some special replacement
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    # lists defined in the begining of the function
    # search for these list of chars and replace it with value in same position 
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # decode emojis as text
    text = emojis.decode(text)
    
    #  each emojis is decoded into :some text:
    text = re.split(r"(:\w+:)+", text)
    
    #  convert text splited to list to string again and encode again the emojis text to emjois icon
    text = ' '.join(text)
    text = emojis.encode(text)

    # Seprate numbers and words
    text = SEP_NUM_WORD.split(text)
    text = ' '.join(text)

    # Seprate english and arabic
    text = SEP_EN_AR.sub(r' \g<0> ', text)

    # remove longation (char repeated more that 2 times)
    text = CHAR_REPEATED.sub(r"\1\1", text)

    # remove more than one space between words
    text = MORE_SPACES.sub(r" ", text)

    return text

########################## End of Clean / Normalize Arabic Tweets


########################## Start TreebankWord Tokenizer

def tokenize_using_nltk_TreebankWordTokenizer(text_list):
    '''
    The function used to segment document into tokens(words), but based on nltk library, 
    but this time using TreebankWordTokenizer which more accurate, 
    and have its own pre-defined rules, and can know different between cases of points
    in end of sentence and in decimal numbers as well as other rules.
    
    Argument
        text_list      : list, list of strings.
    
    Return
        tokenized_text : list, list of lists each of these list are tokens which consist that string.
    '''

    # Take and object from TreebankWordTokenizer tokenizer
    tree_tokenizer = TreebankWordTokenizer()

    # Tokenize each text into tokens
    tokenized_text = [tree_tokenizer.tokenize(text) for  text in text_list]

    return tokenized_text


########################## End TreebankWord Tokenizer


########################## Start get n-grams of tokens

def get_all_ngrams(text_list, nrange=3):
    '''
    The function used to get some n-grams instead of just uni-grams,
    like "We can do it" as uni-gram [We, can, do, it] it will be, [We_can, can_do, do_it] as bi-gram and so on.

    Arguemtn
        text_list : list, list that contain our text each of these text are list of tokens (uni-gram tokens).
    Return
        text_list : list, list that contain our text each of these text are list of tokens along with n-gram tokens.
    '''

    for i, text in enumerate(text_list):
        ngs = []
        # get from uni-gram to n-grams you need for that tweet
        for n in range(1,nrange+1):
            ngs += [ng for ng in ngrams(text, n)]
            # seprate the grams by _ like We_can
            text_list[i] = ["_".join(ng) for ng in ngs if len(ng)>0 ]
    return text_list

########################## End get n-grams of tokens



########################## Start back to original text from tokenized one

def one_text_un_tokenization(text):
    '''
    The function used to back the text into base form instead of tokenized form.

    Argument
        text : list, list of tokens that represent some text.
    return
        text : string, convert the tokenized list into string.
    '''

    text = " ".join(text)
    return text
    
########################## End of back to original text from tokenized one

########################## Start To get lemma for tokens

def one_text_Lemmatizing(text):
    '''
    The function used to get the lemma of the words, 
    one_text_Lemmatizing is the process of mapping all the different forms of a word to its base word,
    but based on the context of the text.

    Argument
        text : string, the text you need to tokenize and get its lemma for each word.
    return
        text : string, the text after getting the lemma of each word its contained.
    '''

    # First we need to tokenize the text into its words (tokens)
    text     = TreebankWordTokenizer.tokenize(text)

    # Get an object of Arabic Lemmitizer
    stemmer  = ArabicLightStemmer()

    # For each word get its base form
    text     = [stemmer.light_stem(word) for word in text]

    # As we need the text again as if we have any other preprocess no the tokenized text, so we untokenize it.
    text     = one_text_un_tokenization(text)

    return text

########################## End of get lemma for tokens


def all_text_Lemmatizing(text_list):
    '''
    The function used to handle multiple of text in one list.

    Argument
        text_list : list, list of strings each of these strings need to get the lemma of its tokens(words).
    return
        text_list : list, list of strings after process and get the lemma of its tokens(words).
    '''

    # Call the one_text_Lemmatizing on each of these text in the list
    text_list = [one_text_Lemmatizing(sentence) for sentence in text_list]

    return text_list



def arabic_preprocess_pipline(file_name_to_read, file_name_to_save, ids_col="id", text_col="text" ,dialect_col="dialect", using_lemma=False, data_path=DATA_PATH):
    '''
    The function used to collect any of the preprocessing functions we need to apply into our data.
    
    Argument
        list_of_text: list, list of text that contain our data text to process it.
    Return
        True, boolean once we have passed through the pipeline without errors
    '''

    # To append text processed in
    text_preprocessed = []

    # Reading the dialect_dataset using read_csv function defined configs file
    new_dialect_dataset  = read_csv(file_name_to_read)

    # Get list of the text columns from our data
    list_of_text = list(new_dialect_dataset[text_col])

    # As we need to process text we need to handle the target instead of chars to numbers,
    # These numbers are labeled from 0 to len(classes), and each map each char to one of these label
    l_encoder = LabelEncoder()
    dialects  = list(new_dialect_dataset[dialect_col])
    l_encoder.fit(dialects)
    dialects = l_encoder.transform(dialects)

    try:
        for text in list_of_text:
            # call clean_text function defined above, and return the cleaned text as list
            text_preprocessed += [clean_text(text)]

        # It takes a time if we run it. call all_text_Lemmatizing defined above.
        if using_lemma:
            text_preprocessed = all_text_Lemmatizing(text_preprocessed)

        #  Display some info about the result
        print("Some samples of text after preprocessed it:")
        print(text_preprocessed[:5])
        print("="*50)


        # Create new dataframe with the retrieve text column as well as with other columns
        dialect_data_frame            = pd.DataFrame({"id": new_dialect_dataset[ids_col], "dialect":  new_dialect_dataset[dialect_col], "dialect_l_encoded":  dialects, "text": text_preprocessed})
        
        # Save as new csv file to start the preprocessing pipeline on
        file_path_to_save = data_path + file_name_to_save
        dialect_data_frame.to_csv(file_path_to_save, index=False, encoding='utf8')
        
        # Check if we missed any instance in our preprocessing pipeline
        assert(len(dialect_data_frame) == len(new_dialect_dataset))

        print("Number of instances in our new file are: ", len(dialect_data_frame))
        print("="*50)
        print("Our Arabic preprocess pipeline is work without any error.")
        
    # Send exception to logs direction
    except Exception as e:
        file                = open("logs/data_preprocess.log","+a")
        file.write("This error related to function arabic_preprocess_pipline of data_preprocess file \n"
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines

    return True