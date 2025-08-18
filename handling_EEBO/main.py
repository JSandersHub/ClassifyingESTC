import extract_EEBO
import multiprocessing
import util
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt

from tqdm import tqdm           # used for a progress bar
from functools import partial   # used for function arguments in the multiprocessing pool
import train_stm

#   This helps with relative file paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tqdm.pandas()

class Config:
    CORE_COUNT = 7 # How many CPU cores to use throughout all functions whenever concurrency is used

    # Find the eebo corpus at https://textcreationpartnership.org/faq/#faq05
    # This must be a directory of all the folders labelled A0, A1, A2, ..., B2, B3, B4
    #   Within each folder, there must be a list of .xml files
    # Please extract both EEBO phase 1 and phase 2 to this same directory
    CORPUS_DIR 				= "../corpus/data"

    # These dates are obtained from the EEBO corpus at https://textcreationpartnership.org/faq/#faq05
    #   Phase 1 and 2 dates are stored under "eebo_phase1" and "eebo_phase2" respectively
    #   Locate the files with the file name "eebo_phase1_IDs_and_dates.txt" and "EEBO_Phase2_IDs_and_dates.txt"
    EEBO_PHASE_1_DATES = "../corpus/eebo_phase1_IDs_and_dates.txt"
    EEBO_PHASE_2_DATES = "../corpus/EEBO_Phase2_IDs_and_dates.txt"

    # This is where the preprocessed files get outputted to
    #   This will contain a large list of .txts of all the text, ready for processing
    PREPROCESSING_OUTPUT 	= "../corpus/preprocessed"

    # This is the file name of the metadata file which gets produced
    #   If you change this, you must also change the train_stm.py globals metadata path to point to the same location
    METADATA_OUTPUT 		= "../corpus/metadata.csv"

    # This is the date range to filter texts by
    #   Date range is inclusive
    START_DATE = 1600
    END_DATE = 1699

    # These are the topic counts used to produce models to compare their suitability (using heldout likelihood)
    K_CANDIDATES = [5, 10, 15] 

    # How many iterations to carry out the model training process
    MAX_EM_ITER = 20 




def get_dates_df() -> pd.DataFrame:
    """Gets a dataframe of all EEBO ids with their respective date

    Returns:
        pd.DataFrame: With column "date". index is EEBO id
    """
    
    def parse_date(date : str) -> int:
        """
        Some dates are presented as a date range, 
            E.g 1695-1696?
            If so I will: take the earliest date and remove the question mark
        They may also contan a "u" (unknown?),
            E.g 169u
            If so I will: replace the "u" for 0
        """
        date = date.replace("?", "").replace("u", "0")
        return int(date.split("-")[0])
    


    # The date files are given as two different tables for phase 1 and phase 2 EEBO
    column_names = ["id", "date"]
    f1 = pd.read_csv(Config.EEBO_PHASE_1_DATES, delimiter="	", names=column_names, index_col=0)
    f2 = pd.read_csv(Config.EEBO_PHASE_2_DATES, delimiter="	", names=column_names, index_col=0)


    # Combine the tables and parse the date information
    df = pd.concat([f1, f2])
    df["date"] = df["date"].apply(parse_date)
    
    return df


def get_files_to_process(directory : str) -> pd.DataFrame:
    """ Grabs all the file paths and ids of each file from the corpus directory

    Returns:
        pd.DataFrame : Pandas dataframe with index column as the file ids, (E.g A04015), and a column "file_path".
    """
    out = []

    for folder in util.loop_directory(directory):
        if not os.path.isdir(folder): continue

        for xml_file in util.loop_directory(folder):


            # os.path.split(xml) file gives ('../corpus/data\\A7', 'A75689.P4.xml')
            #   We want the id from this file name
            path, filename = os.path.split(xml_file)
            id = filename.split(".")[0]
            out.append({
                "id" : id,
                "file_path" : xml_file,
            })

    return pd.DataFrame(out).set_index("id")



def get_file_metadata(file_data : tuple[str, str]) -> pd.Series:
    """Given the file data of an EEBO file, return a dictionary of any useful data extracted
    The extract_EEBO.EEBO_parser does the heavy lifting here

    I made this function with the purpose of being compatible with multiprocessing.Pipe.imap()

    This "useful data" will be eventually go on to be saved with the metadata.csv for STM training

    Args:
        file_data tuple[str, str]: Tuple of the form (EEBO_ID, FILE_PATH). E.g ("A50000", "corpus/data/A5/A50000.P4.xml")
    """
    id, file_path = file_data
    parser = extract_EEBO.EEBO_parser(file_path)

    out = {
        "id"        : id,
        "title" 	: preprocess_text(parser.get_title()),
        "author" 	: preprocess_text(parser.get_authors()),
        "publisher" : preprocess_text(parser.get_publisher()),
        "pubplace" 	: preprocess_text(parser.get_pubplace()),
        "textclass" : preprocess_text(parser.get_textclass()),
    }

    return pd.Series(out)

def get_all_files_metadata(file_df : pd.DataFrame) -> pd.DataFrame:
    """Given a list of files from file_df (under column "file_path"), create the metadata dataframe, extracting data from each file
    """

    # Now extract metadata from all the files
    with multiprocessing.Pool(processes=Config.CORE_COUNT) as pool:
        try:
            results = list(tqdm(pool.imap(get_file_metadata, file_df["file_path"].items()), total=len(file_df), desc="Metadata"))
        except KeyboardInterrupt as e:
            pool.terminate()

    results = pd.DataFrame(results).set_index("id")
    return results


def preprocess_text(text : str) -> str:
    """Given a text string, preprocess (remove stop words, spaces, numbers, and stem all words)
    """
    stopwords = util.Utils.get_stopwords()
    stemmer = util.Utils.get_stemmer()

    # Convert to lowercase
    text = text.lower()

    # Keep just characters, spaces and numbers (soucrce 00-estc_btm_prep.ipynb)
    text = re.sub(r'[^a-zA-Z\s]+', '', text)

    # Tokenise
    tokens = word_tokenize(text)

    # Filter out all stop words and stem every word
    filtered_words = [stemmer.stem(word) for word in tokens if word not in stopwords and len(word) >= 2]

    # Collapse back together
    return " ".join(filtered_words)




# Bulk processing

def preprocess_and_save_file(file_path, output_dir : str) -> None:

    # Open file and parse it
    parser = extract_EEBO.EEBO_parser(file_path)

    # Extract the parsed text and preprocess it
    content = preprocess_text(parser.get_parsed_text())

    # Write the preporcessed content to a new file in the preprocessed directory
    path, name = os.path.split(file_path)
    with open(os.path.join(output_dir, name+f".txt"), "w") as f:
        f.write(content)


def save_and_preprocess_all_files(output_dir : str, file_df : pd.DataFrame):
    """
    Call this function to start preprocessing all texts in the corpus

    Args:
        output_dir : The directory to output all the texts after preprocessing
        file_df : A dataframe containing the column "file_path". All file paths are read and preprocessed from this dataframe
    """
    files = file_df["file_path"]

    process_function = partial(preprocess_and_save_file, output_dir=output_dir)

    with multiprocessing.Pool(processes=7) as pool:
        try:
            results = list(tqdm(pool.imap(process_function, files), total=len(files), desc="Preprocessing"))
        except KeyboardInterrupt as e:
            pool.terminate()


def preprocess_main():
    if not os.path.exists(Config.METADATA_OUTPUT):
        print("METADATA file not found, regenerating... ")

        # Get the dataframe containing all the file ids with their date
        date_df = get_dates_df()    

        # Get the datafram containing all the file paths with their id
        file_df = get_files_to_process(Config.CORPUS_DIR)

        # Combine the date df with the metadata df
        file_df["date"] = date_df["date"]

        # Filter out all unwanted dates
        file_df = file_df[(file_df["date"] <= Config.END_DATE) & (file_df["date"] >= Config.START_DATE)]


        metadata = get_all_files_metadata(file_df)

        file_df = file_df.join(metadata)
        file_df.to_csv(Config.METADATA_OUTPUT)
    else:
        file_df = pd.read_csv(Config.METADATA_OUTPUT)

    print("Saving data distribution to './img/date_distribution.png'")
    file_df.hist("date")
    plt.savefig("./img/date_distribution.png")


    save_and_preprocess_all_files(Config.PREPROCESSING_OUTPUT, file_df)

def create_corpus_main():
    train_stm.create_corpus(Config.CORE_COUNT)

def model_fitting_main():
    train_stm.start_reference_model_fitting(Config.K_CANDIDATES, Config.CORE_COUNT, Config.MAX_EM_ITER)

def main():
    def display_options(options : dict):
        print("\n\n\nAvailable commands:")
        for option in options:
            print(f"\t{option}) {options[option]['message']}")
        print("\n\n\n")


    MENU_OPTIONS = {
        "1" : {"func": preprocess_main, "message" : "Preprocess: Create metadata csv if doesnt exist, filters all the dates, and preprocesses all files in the corpus."},
        "2" : {"func" : create_corpus_main, "message" : "Create BOW Corpus: Create the BOW corpus for the STM training"},
        "3" : {"func" : model_fitting_main, "message" : "Train Reference Models: Start training the model from the corpus produced in option 2"},
        "4" : {"func" : train_stm.plot_heldout, "message" : "Plotting Heldout Likelihood: Show and save the graph of heldout likelihood against topic count, using data from option 3"},
        "5" : {"func" : train_stm.scrutinise_model, "message" : "Model Scrutinisation: Train a new model using whole BOW corpus from option 2, but use optimal topic count decided from option 3. Save and display results"},
        "e"	: {"func": None, "message": "Exit program"},
    }

    while True:
        display_options(MENU_OPTIONS)

        func = input("Please choose: ").lower()

        if func not in MENU_OPTIONS.keys(): continue

        if func == "e" : return

        MENU_OPTIONS[func]["func"]()

    


if __name__ == "__main__":
    main()






# held out likelihood