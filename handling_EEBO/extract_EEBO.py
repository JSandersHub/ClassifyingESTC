# This file contains all the relevant functions and config towards handling the data provided from the EEBO
#   That is more specifically, containing the parsers and functions to read the files to extract text from the xml files

import os
import xml.etree.ElementTree as xet

# Set the CWD of python file to the location the python file is installed
#   This helps with relative file paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class XMLParsingError(Exception):
    def __init__(self, message):
        self.message = message


class EEBO_parser:
    class UnrecognisedFileType(Exception):
        def __init__(self, message):
            self.message = message


    def __init__(self, file_path : str):
        self.file_path = file_path

        self.__raw_text 	= None
        self.__parsed_text 	= None
        self.__xml_tree 	= None

    def __extract_parsed_text(self) -> str:
        root = self.get_xml_tree()

        # As far as I know, the header information can be consistently ignored in all these files
        #   Hopefully, they all follow this scheme where the relevant content of the file is contained in the <EEBO> tag
        text = root.find("./EEBO")

        out = ""
        for line in text.itertext():
            """
            Removes cases where a word is cut off mid-line and trails to the next sentence
                e.g
                I am doing wonder-
                ful this evening.

                Many thanks.

                ->

                I am doing wonderful this evening. Many thanks. 
            """
            out += line.replace("-\n", "").replace("\n", " ")
        
        return out
    

    def __extract_raw_text(self) -> str:
        if self.__raw_text is None:
            file_path = self.file_path
            root, ext = os.path.splitext(file_path)

            if ext == ".xml":
                with open(file_path, "r", encoding="utf-8") as xml_file:
                    return xml_file.read()

            else:
                raise self.UnrecognisedFileType(f"Incorrect file type {file_path}. Must be .xml")
            

    def get_xml_tree(self) -> xet.Element:
        if self.__xml_tree is None:
            self.__xml_tree = xet.fromstring(self.get_raw_text())

        return self.__xml_tree


    def get_raw_text(self) -> str:
        """Extracts the raw text body (text including XML) of the file and returns it.

        Raises:
            UrecognisedFileType: If the given file path is not a .xml or .gz

        Returns:
            str: The text INCLUDING xml data
        """
                
        if self.__raw_text is None:
            self.__raw_text = self.__extract_raw_text()
        return self.__raw_text
    
    def get_parsed_text(self) -> str:
        """Parses xml text from the EEBO corpus

        Returns:
            str: The text body (without XML tags all on one line)
        """

        if self.__parsed_text is None:
            self.__parsed_text = self.__extract_parsed_text()
        return self.__parsed_text
    
    def get_text_from_element(self, xml_path : str, graceful = True) -> str | None:
        root = self.get_xml_tree()
        elem = root.find(xml_path)

        if elem is None:
            if graceful: return None
            raise XMLParsingError(f"Unknown path {xml_path} for file {self.file_path}")
        

        return elem.text

    def get_title(self) -> str:
        return self.get_text_from_element("./HEADER/FILEDESC/SOURCEDESC/BIBLFULL/TITLESTMT/TITLE")
    
    def get_publisher(self) -> str:
        return self.get_text_from_element("./HEADER/FILEDESC/SOURCEDESC/BIBLFULL/PUBLICATIONSTMT/PUBLISHER")
    
    def get_pubplace(self) -> str:
        return self.get_text_from_element("./HEADER/FILEDESC/SOURCEDESC/BIBLFULL/PUBLICATIONSTMT/PUBPLACE")
    
    def get_authors(self) -> str:
        root = self.get_xml_tree()

        authors = []
        for author in root.findall("./HEADER/FILEDESC/SOURCEDESC/BIBLFULL/TITLESTMT/AUTHOR"):
            authors.append(author.text)
    
        return "; ".join(authors)
    
    def get_textclass(self) -> str:
        root = self.get_xml_tree()

        keywords = []
        for keyword in root.findall("./HEADER/PROFILEDESC/TEXTCLASS/KEYWORDS/TERM"):
            keywords.append(keyword.text)
    
        return "; ".join(keywords)


        


if __name__ == "__main__":
    extractor = EEBO_parser(r"..\corpus\data\A6\A69655.P4.xml")
    date = extractor.get_textclass()
    print(date)