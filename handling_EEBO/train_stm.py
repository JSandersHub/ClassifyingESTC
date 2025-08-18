# This is adapting the code found in the repo https://github.com/mkrcke/strutopy/tree/main
#   To work with the EEBO dataset

import os
import re
import numpy as np
import pandas as pd
from gensim import corpora

import util
import multiprocessing
import random

from tqdm import tqdm 
from functools import partial

from strutopy.modules.stm import STM
from strutopy.modules.heldout import eval_heldout

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


#   This helps with relative file paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Config:
    METADATA_PATH		= "../corpus/metadata.csv"
    BOW_OUTPUT			= f"../corpus/stm/BoW_corpus-small.mm"
    BOW_ID_MAPPING		= f"../corpus/stm/bow_mapping.csv"
    PREPROCESSED_DIR	= "../corpus/preprocessed"

    REFERENCE_MODEL_OUTPUT =  "../corpus/stm/eebo_data/"
    _METADATA = None # Singleton for all objects to get metadata using get_raw_metadata(0)



def get_raw_metadata():
    if Config._METADATA is None:
        print(f"Reading in metadata from {Config.METADATA_PATH}")
        Config._METADATA = pd.read_csv(Config.METADATA_PATH)
    return Config._METADATA

def get_metadata_with_bow_mapping() -> pd.DataFrame:
    """Returns the metadata file as a pandas df but merged with the id's of the corresponding Bag of Word document
    This is needed because the BoW corpus does not support custom id, so I cannot assign them the ID of the EEBO corpus (e.g A00145)

    So instead, I have another file called bow_mapping.csv" that links the two id formats together
    This function merges the metadata.csv with this mapping, which enables the STM to use the BoW ids to reference the metdata
    """

    metadata = get_raw_metadata()

    # This maps the metadata from the preporcessing to the BoW corpus documents
    print(f"Joining metadata with {Config.BOW_ID_MAPPING}")
    bow_mapping = pd.read_csv(Config.BOW_ID_MAPPING)
    metadata = pd.merge(bow_mapping, metadata, left_on="id", right_on="id")

    return metadata



# Step 1

def read_file_to_bow(file_path : str, dictionary : corpora.Dictionary):
    """Thie point of this function is to be compatibile with the python multiprocessing.Pipe

    Each preprocessed file will be passed to here, where it will be split into a bag of words
    This bag of words will be added to a list after the function call, which will then be saved
    This acts as the main corpus used in STM training

    Args:
        file_path (str): Path to a preprocessed EEBO file, containing just words
        dictionary (corpora.Dictionary): Dictionary to update
    """

    with open(file_path, "r") as f:
        return dictionary.doc2bow(f.read().split(), allow_update =True)

# WARNING: This function takes up a lot of memory
#   Approximately 12GB+ free space would be needed
def create_corpus(core_count):
    """_summary_

    Args:
        core_count (_type_): _description_
    """

    # This function is adapted from \strutopy\02_create_corpus.py

    print("WARNING: This function uses up a lot of memory (Approximately 12GB with whole corpus)")
    choice = input("Do you wish to proceed? (y/n): ").lower()

    if choice != "y": return

    dictionary = corpora.Dictionary()

    print("Creating BOW")

    process_function = partial(read_file_to_bow, dictionary=dictionary)
    file_batch = list(util.loop_directory(Config.PREPROCESSED_DIR))

    # Get user input to choose sample size
    F_COUNT = len(file_batch)
    SAMPLE_SIZE = int(input(f"Input how many files to sample from {F_COUNT} files: "))
    SAMPLE_SIZE = min(F_COUNT, SAMPLE_SIZE)
    print(f"Sampling {SAMPLE_SIZE}/{F_COUNT} files ( {100*SAMPLE_SIZE/F_COUNT:.2f}% )")

    file_batch = random.sample(file_batch, SAMPLE_SIZE)

    with multiprocessing.Pool(processes=core_count) as pool:
        try:
            BoW_corpus = list(tqdm(
                    pool.imap(process_function, file_batch),
                    total=len(file_batch), 
                    desc="BOW"
                ))
        except KeyboardInterrupt as e:
            pool.terminate()

    # Map id to file
    #   when saving the BOW, the index starts at 1
    output = []
    for i, file_path in enumerate(file_batch):
        path, name = os.path.split(file_path)
        id = name.split(".")[0]
        output.append({"id" : id, "bow_mapping": i+1})

    pd.DataFrame(output).set_index("bow_mapping").to_csv(Config.BOW_ID_MAPPING)

    print(f"Saving BOW corpus to {Config.BOW_OUTPUT}...")
    corpora.MmCorpus.serialize(Config.BOW_OUTPUT, BoW_corpus)







# Step 2

def train_on_corpus(K, beta_train_corpus, theta_train_corpus, heldout_corpus, metadata : pd.DataFrame, max_em_iter, sigma_prior, convergence_threshold):
    # This is taken from handling_EEBO\strutopy\06_example_application.py
    """
    1) Iterate over candidates for the number of topics K
          - choose K from [5,10,20,30,40,50]
    2) Fit the model on the training data (beta and theta separately)
          - a separate fit for theta and beta resembles the document completion approach.
          - create a corpora for training on the train and the train+first_test_half respectively.
          - models not necessarily need to converge, max EM iterations are set to 25
          - both model parameters are stored
    3) Evaluate the heldout likelihood on the test data
          - values obtained from step (2) are used to evaluate the heldout likelihood on second_test_half
          - likelihood is stored for comparing different model fits
    """
    stm_config = {
        "model_type": "STM",
        "content": False,
        "K": K,
        "kappa_interactions": False,
        "sigma_prior": sigma_prior,
        "convergence_threshold": convergence_threshold,
        "init_type": "spectral",
        "max_em_iter": max_em_iter
    }

    print(f"Fit STM on the reference corpus assuming {K} topics")

    
    results_path = os.path.join(Config.REFERENCE_MODEL_OUTPUT, f"STM_{K}/")
    print(f"Creating directory {results_path}")
    os.makedirs(results_path, exist_ok=True)


    # initialize models for theta and beta

    corpus = [doc for doc_id, doc in beta_train_corpus]
    X = metadata.loc[[doc_id for doc_id, doc in beta_train_corpus]].values

    model_beta = STM(
        documents	= corpus,
        dictionary	= corpora.Dictionary.from_corpus(corpus),
        X			= X,
        **stm_config,
    )

    corpus = [doc for doc_id, doc in theta_train_corpus]
    X = metadata.loc[[doc_id for doc_id, doc in theta_train_corpus]].values

    model_theta = STM(
        documents	= corpus,
        dictionary	= corpora.Dictionary.from_corpus(corpus),
        X			= X,
        **stm_config,
    )
    
    # Train model to retrieve beta and theta (document completion approach)
    print(f"Fitting STM for K={K} ...")
    
    model_beta.expectation_maximization(saving=True, output_dir=results_path)
    model_theta.expectation_maximization(saving=True, output_dir=results_path)

    # Save Likelihood
    print(f"Evaluate the heldout likelihood on the remaining words...")
    heldout_corpus = [doc for doc_id, doc in heldout_corpus]
    heldout_llh = eval_heldout(
        heldout_corpus, theta=model_theta.theta, beta=model_beta.beta
    )
    print(f"Saving into {results_path}.")
    
    heldout_path= os.path.join(results_path,"heldout")
    np.save(heldout_path, np.array(heldout_llh))


def start_reference_model_fitting(K_candidates, core_count, max_em_iter):

    # This function is taken from handling_EEBO\strutopy\06_example_application.py

    print(f"Producing model for topic sizes {K_candidates}")
    choice = input("Do you wish to proceed? (y/n): ").lower()
    if choice != "y": return


    metadata = get_metadata_with_bow_mapping()
    metadata = metadata[["date","title","author","publisher","pubplace","textclass"]]

    BoW_corpus = corpora.MmCorpus(Config.BOW_OUTPUT)

    print(f"Size of corpus: {len(BoW_corpus)}")

    # split corpus based on 60/40 train-test split
    TRAIN_TEST_PROPORTION = 0.6

    print("Splitting corpus...")

    corpus			 = [(bow_id, doc) for bow_id, doc in enumerate(BoW_corpus)]
    corpus_split_idx = int(TRAIN_TEST_PROPORTION * len(BoW_corpus))

    beta_train_corpus = corpus

    test_docs 		= corpus[corpus_split_idx:]
    test_split_idx 	= len(test_docs) // 2

    first_half, second_half	= test_docs[:test_split_idx], test_docs[test_split_idx:]
    heldout_corpus			= first_half
    theta_train_corpus		= corpus[:corpus_split_idx] + second_half


    # Prepare corpora for model training
    print("Creating training + heldout corpora")

    # %% Fit the model for K candidates and save the results

    print("Starting training..")

    # specify model parameters
    sigma_prior = 0
    convergence_threshold = 1e-5    

    print(metadata)

    process_function = partial(
        train_on_corpus, 
        beta_train_corpus		= beta_train_corpus,
        theta_train_corpus		= theta_train_corpus,
        heldout_corpus			= heldout_corpus,
        metadata				= metadata,
        max_em_iter				= max_em_iter,
        sigma_prior				= sigma_prior,
        convergence_threshold	= convergence_threshold
    )

    core_count = 1
    with multiprocessing.Pool(processes=min(core_count, len(K_candidates))) as pool:
        print("Starting processing pool...")
        results = list(pool.imap(process_function, K_candidates))






# Step 3

def plot_heldout():
    
    #%% Evaluate heldout likelihood for candidate models
    def list_files(filepath, filetype):
        paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.lower().endswith(filetype.lower()):
                    paths.append(os.path.join(root, file))
        return(paths)

    path = Config.REFERENCE_MODEL_OUTPUT
    result_paths = list_files(path, "heldout.npy")

    print(f"Comparing results from: {result_paths}")

    values=[]
    K=[]
    for path in result_paths: 
        match = re.search(r"STM_(\d*)", path)
        k_value = int(match.group(1))
        K.append(k_value)

        heldout = np.load(path)
        values.append(heldout)

    result_frame = pd.DataFrame(
        {
            "K":K,
            "heldout":np.round(values,3)
        }
    )

    BoW_corpus = corpora.MmCorpus(Config.BOW_OUTPUT)


    ax = sns.lineplot(data=result_frame.sort_values(by="K"), x="K", y="heldout")
    ax.scatter(data=result_frame.sort_values(by="K"), x="K", y="heldout", c="#FF0000", s=4)
    ax.set(xlabel="Number of topics", ylabel="Per Word Heldout Likelihood")
    ax.set_title(f"Heldout likelihood against number of topics for corpus size N = {len(BoW_corpus)}")

    print("Saving heldout likelihood comparison to ./img/heldout_likelihood")
    plt.savefig("./img/heldout_likelihood.png")
    plt.show()
    
    # For a corpus size of 421, it appears that K = 45 topics is optimal (i.e highest point in range 10 < k < 150)






# Step 4

def get_preprocessed_from_id(ids):
    def file_path_to_preprocessed(file_path):
        prep_path = Config.PREPROCESSED_DIR
        path, name = os.path.split(file_path)

        new_path = os.path.join(prep_path, name+".txt")
        with open(new_path, "r") as f:
            return f.read()
    metadata = get_metadata_with_bow_mapping()

    return metadata["file_path"].iloc[ids].apply(file_path_to_preprocessed)



def scrutinise_model():
    K = int(input(f"Input optimum topic count: "))

    MODEL_PATH = os.path.join(Config.REFERENCE_MODEL_OUTPUT, f"STM_{K}/")

    if not os.path.exists(MODEL_PATH):
        print(f"File for topic count does not exist (searching for {MODEL_PATH})")
        return
    

    print("Reading in BOW corpus...")
    BoW_corpus = corpora.MmCorpus(Config.BOW_OUTPUT)
    print("Generating dictionary from corpus...")
    dictionary = corpora.Dictionary.from_corpus(BoW_corpus)
    metadata = get_metadata_with_bow_mapping()


    stm_config = {
        "model_type"			: "STM",
        "content"				: False,
        "K"						: K,
        "kappa_interactions"	: False,
        "lda_beta"				: True,
        "sigma_prior"			: 0,
        "convergence_threshold"	: 1e-6,
        "init_type"				: "spectral",
        "max_em_iter" 			: 200,
    }
    
    print(f"Creating new model with config: {stm_config}")

    scrutinised_model = STM(
        documents=BoW_corpus,
        dictionary=dictionary,
        X=metadata.values,
        **stm_config,
    )

    print("Starting training...")

    scrutinised_model.expectation_maximization(saving=False)

    #
    # Past this point, I don't necesserily understand what is happening - PLEASE VERIFY FOR CORRECTNESS
    #   Please read where I took this code from : https://github.com/mkrcke/strutopy/blob/main/src/06_example_application.py
    # I simply just adapted this to work with our corpus and metadata
    #

    #%% investigate advantage of spectral decomposition

    lb_spectral = np.load(MODEL_PATH+"lower_bound.pickle", allow_pickle=True)
    lb_random = scrutinised_model.last_bounds # here scrutinised_model was estimated with "init_type": "random"

    plt.plot(lb_random, label = "random initialisation")
    plt.plot(lb_spectral, label = "spectral initialisation")
    plt.xlabel("Number of em iterations")
    plt.ylabel("Lover bounds")
    plt.legend()

    print("Saving comparison results to ./img/spectral_initialisation.png")
    plt.savefig("./img/spectral_initialisation.png", dpi=360, bbox_inches="tight")


    # %% investigate topics (highest probable words)
    K=10
    prob, frex = scrutinised_model.label_topics(n=10, topics=range(K))
    # investigate covariate effect on topics
    for topic in range(K): 
        print(f"Statistics: {round(scrutinised_model.gamma[topic][0],4)} * {frex[topic]})")
        print(f"ML:  {round(scrutinised_model.gamma[topic][1],4)} * {frex[topic]} \n")
    # %%
    _,frex = scrutinised_model.label_topics(n=10, topics=range(0,10))
    for i,topic in enumerate(frex):
        print('-'*130)
        print(f"Topic {i+1}:",topic[1:])

    # %% investigate representative documents per topic
    # The function find_thoughts() can be used to retrieve the most representative documents
    # for particular, indexed topics. 
    topics = [0]
    for topic in topics: 
        print('Documents exhibiting topic "Testing"')
        titles = metadata["title"].iloc[scrutinised_model.find_thoughts(n=20, topics=[topic])]
        for i,title in enumerate(titles):
            print(f'document {i} {title}')

        
    # %% investigate representative documents for (a single) topic 0 
    topic = 5
    # %% topics with overlap
    # to find topics that overlap, extract values where both gammas are of equal sign
    stats_topics = np.where(scrutinised_model.gamma[:,0]>0)[0]
    ml_topics = np.where(scrutinised_model.gamma[:,1]>0)[0]

    # ML: label_topics
    # Statistics: label_topics
    for topic in ml_topics: 
        scrutinised_model.label_topics(n=15, topics =[topic], print_labels=True)
    # Statistics: label_topics
    for topic in stats_topics: 
        scrutinised_model.label_topics(n=15, topics =[topic], print_labels=True)

    #%% for each topic, find representative documents
    for topic in ml_topics:
        print('-'*25)
        print(np.array(metadata[['title']].iloc[scrutinised_model.find_thoughts(n=10, topics=[topic])]))

    #%% Statistics
    for topic in stats_topics:
        print('-'*25)
        print(np.array(metadata[['title']].iloc[scrutinised_model.find_thoughts(n=10, topics=[topic])]))

    # %% distinct topics
    # get largest absolute the distances in the covariate effects for five topics
    #compute the absolute differences
    gamma_diff = abs(scrutinised_model.gamma[:,1]-scrutinised_model.gamma[:,0])
    distinct_topics = np.argpartition(gamma_diff, -5)[-5:]
    # to find topics that are distinct, extract values where differences in gamma is largest
    # label_topics
    for topic in distinct_topics: 
        scrutinised_model.label_topics(n=15, topics =[topic], print_labels=True)
    # find_thoughts

    # %%
    # wordclouds:
    # 0) Generate a wordcloud for the entire data
    # 1) extract representative documents for topic representatives
    # 2) extract texts from the original data for the representatives
    # 3) display words using wordcloud

    # 1) generate a wordcloud for the whole corpus

    stopwords = util.Utils.get_stopwords()

    x, y = np.ogrid[:300, :300]
    flat_corpus = [token for text in get_preprocessed_from_id([i for i in range(0, len(BoW_corpus))]) for token in text.split()]
    wc = WordCloud(max_words=1000, stopwords=stopwords, margin=5,
                random_state=1, background_color="white").generate(" ".join(flat_corpus))

    plt.imshow(wc)
    plt.axis("off")
    plt.show()
    plt.clf()


    # 2) generate a wordcloud for selected topics (2 x ML, 2 x Statistics)
    ##### STATISTICS CLOUDS #######
    for topic in stats_topics:
        topic_texts = np.array( get_preprocessed_from_id(scrutinised_model.find_thoughts( n=30, topics=[topic] )) )
        flat_topic_texts = [token for sublist in topic_texts for token in sublist.split()]
        wc = WordCloud(max_words=1000, stopwords=stopwords, margin=5,
                random_state=1, background_color="white").generate(" ".join(flat_topic_texts))
        plt.imshow(wc)
        plt.axis("off")
        plt.savefig(f'./img/stats_{topic}', bbox_inches='tight', dpi=400)
        plt.clf()

    ##### MACHINE LEARNING CLOUDS #######
    for topic in ml_topics:
        topic_texts = np.array( get_preprocessed_from_id(scrutinised_model.find_thoughts( n=30, topics=[topic] )) )
        flat_topic_texts = [token for sublist in topic_texts for token in sublist.split()]
        wc = WordCloud(max_words=1000, stopwords=stopwords, margin=5,
                random_state=1, background_color="white").generate(" ".join(flat_topic_texts))
        plt.imshow(wc)
        plt.axis("off")
        plt.savefig(f'./img/ml_{topic}', bbox_inches='tight', dpi=400)
        plt.show()
        plt.clf()






if __name__=="__main__":
    # create_corpus()
    # start_reference_model_fitting()
    # plot_heldout()
    scrutinise_model()
    # print(get_preprocessed_from_id([1,2,3,4]))

    # BoW_corpus = corpora.MmCorpus(Config.BOW_OUTPUT)

    # dictionary = corpora.Dictionary.from_corpus(BoW_corpus)
    # dictionary.filter_extremes(no_below=10, no_above=0.5)

    # print(len(dictionary))

    # pass