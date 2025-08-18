import os
from typing import Generator

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class Utils:
    __stopwords = None
    __stemmer = None

    @classmethod
    def get_stopwords(self) -> set:
        if self.__stopwords is None:
            """
                From 00-estc_btm_prep.ipynb
            """

            # Base stopwords from nltk
            base_stopwords = stopwords.words("english")

            # Additional stopwords from analysing intermediate outputs
            additional_stopwords = [
                "also", 
                "ad",
                "may",
                "upon",
                "unto",
                "one",
                "sic",
                "haue",
                "thou",
                "shall",
                "ye",
                "may",
                "v",
                "thy"
            ]

            # Combine stopwords
            self.__stopwords = set(base_stopwords + additional_stopwords)

        return self.__stopwords
    

    @classmethod
    def get_stemmer(self) -> PorterStemmer:
        """
            From 00-estc_btm_prep.ipynb
        """
        if self.__stemmer is None:
            
            # Initialise stemmer
            self.__stemmer = PorterStemmer()

        return self.__stemmer
    

def display_progress(current_progress : int, max_progress : int, message : str, step=20):
    """Displays a percentage progress on the same like, without output to next
            Intended to be used with a for loop

        Include print() after the for loop to keep the message there without being replaced.

        Args:
            message : Message to display before the percentage
            step : Display the message every N progress increase
    """
    if current_progress % step == 0:
        print(f"{message}{current_progress*100 / max_progress:.2f}%", end="\r")


def loop_directory(directory : str) -> Generator[None, str, None]:
    for folder in os.listdir(directory):
        yield os.path.join(directory, folder)
