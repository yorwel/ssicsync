"""
Ingestion of AR reports pdf files
Creation Date: 14-Jul-2024
Last Modified Date: 14-Jul-2024
@author: Wee Yang

Ingestion of PDF files
"""
# Import packages
import os, nltk, re, string, pickle
import unicodedata as ucd
import pdfplumber as pp

# Update nltk dictionaries
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Import functions
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from datetime import datetime

# Load global variables
stop = stopwords.words('english')

# Pre-process text
def transformText(txt): 
    # Tokenise passed text
    tokens = word_tokenize(txt)
    
    # Remove punctuation
    tokens_nop = [t for t in tokens if not t in string.punctuation]
    
    # Lower case
    tokens_lower=[t.lower() for t in tokens_nop]
    
    # Remove stop words
    tokens_nostop=[t for t in tokens_lower if t not in stop]
    
    # Remove numbers
    tokens_nonum = [t for t in tokens_nostop if not t.isnumeric()]  # standard numbers
    tokens_nonum = [t for t in tokens_nonum if re.match('^\$?\d+(\.\d{2})?$', t) is None]   # currency, no commas allowed
    tokens_nonum = [t for t in tokens_nonum if re.match('^\d{1,3}(,\d{3})*(\.\d+)?$', t) is None]   # comma-grouped, between powers of 1,000
    tokens_nonum = [t for t in tokens_nonum if re.match('^([1-9]\d*|0)(\.\d+)?$', t) is None]   # leading and trailing zeros
    
    # Trim words
    tokens_trimmed = [t.strip() for t in tokens_nonum]  # whitespaces in front or back
    tokens_trimmed = [re.sub('\\d*', '', t) for t in tokens_trimmed]    # removes any numbers in front or back
    tokens_trimmed = [re.sub('^\/*|\/*$|^\.*|\.*$|^,|,$|^-|-$', '', t) for t in tokens_trimmed]    # removes any residue leading/trailing punctuation: /.,-
    
    # Removes elements with certain attributes
    tokens_final = [t for t in tokens_trimmed if len(t) > 0]    # zero length
    tokens_final = [t for t in tokens_final if re.match("'", t) is None]    # contains apostrophe "'s" etc
    tokens_final = [t for t in tokens_final if len(t) > 1 or not ucd.category(t).startswith('P')] # more extensive unicode library that includes artistic punctuation
    tokens_final = [t for t in tokens_final if len(t) > 1 or not ucd.category(t).startswith('S')] # math symbols
    
    # Function return
    return tokens_final

# Filter pages with Notes Financial Statements
def captureNotesFinancialStatementsPages(txt): 
    # Search for regex match
    result = re.search('notes?:?\s?(to)?\s?(the)?\s?financial\s?statements?', txt.lower())
    
    if result is not None: 
        # Search for the term "principle activites" or similar
        pAct = re.search('principal activit(ies|y)', txt.lower())
        
        if pAct is not None: 
            # Look for new line characters to determine POS start
            startIndex = [m.end() for m in re.finditer('\n', txt[:pAct.span()[0]])]
            
            # Look for period to determine POS send
            endIndex = re.search('\.', txt[pAct.span()[1]:]).span()[0]
            
            # Return POS that contains principal business activity
            return txt[startIndex[len(startIndex) - 1]:pAct.span()[1] + endIndex] 
        else: 
            return None
        
    else: 
        return None

# Filter pages with website
def captureWebsitePages(txt): 
    # Search for regex match
    result = re.search('((https?|ftp|smtp):\/\/)?(www.)?[a-z0-9]+\.[a-z]+(\.[a-z]+)?', txt)
    
    # Regex used would confused with email
    email = re.search('[a-z0-9]+\.?([a-z0-9]+)?@([a-z0-9]+)(\.[a-z0-9]+)?(\.[a-z0-9]+)?', txt)
    
    # Excludes email segments being erroneously picked up by website regex
    if result is not None: 
        if email is None: 
            return result[0] if len(result[0]) > 5 else None     # may capture "e.g" or very short tokens
            
        else: 
            resultSpan = result.span()
            emailSpan = email.span()
            
            if resultSpan[1] <= emailSpan[0] or resultSpan[0] >= emailSpan[1]:
                return result[0] if len(result[0]) > 5 else None     # may capture "i.e" or very short token
            else: 
                return None
    else: 
        return None

# Extract one single page text
def extractInfo(obj): 
    # Initialise an empty dictionary
    main = dict()
    website = dict()
    
    # Empty string variable to store website address
    t = ''
    
    # Loop thru all pages
    # for pg in obj.pages: 
    #     txt = pg.extract_text()
    #     exists = captureNotesFinancialStatementsPages(txt)
        
    #     # Search for Notess to Financial Statements
    #     if exists is not None: 
    #         main.update({'pg' + str(pg.page_number): txt})
            
    #     exists = captureWebsitePages(txt)
        
    #     if exists is not None: 
    #         website.update({'pg' + str(pg.page_number): exists})
    
    for i in range(len(obj.pages)):  
        # Search for Notes to Financial Statements
        exists = captureNotesFinancialStatementsPages(obj.pages[i].extract_text())
        
        # Prevents any empty list (bcos empty page) from being included
        if exists is not None: 
            # Combine with main dictionary
            main.update({'page{}'.format(str(i+1)): exists})
            
        # Search for company website
        exists = captureWebsitePages(obj.pages[i].extract_text())
        
        if exists is not None: 
            if len(exists) > len(t):    # longest found string that indicates website
                t = exists
                
            # Combine with website dictionary
            website.update({'page{}'.format(str(i+1)): t})
        
    # Return combined main dictionary
    return main, website

# Stitch into one dictionary
def PDFWrangling(home, files): 
    # Initialise an empty dictionary
    main = dict()
    websites = dict()
    
    # Loop thru all available files
    for f in files: 
        # Extract text dictionary from object
        print('Extracting {}... {}'.format(f, datetime.now()))
        
        with pp.open(os.path.join(rptPath, f)) as arFile:
            notes, website = extractInfo(arFile)
        
        # Combine into main dictionary with AR filename and note pages
        main.update({f: notes})
        
        # Combine into website dictionary with AR filename and website pages
        websites.update({f: website})
        
        print('File {} extraction completed! {}'.format(f, datetime.now()))
        
    # Return dictionary
    return main, websites

# Merge all pages of files into one big list
def mergePages(dct): 
    # Initialise an empty dictionary
    main = dict()
    
    # Loop through all files
    for ka, va in dct.items():
        # Initialise an empty consolidated list
        lst = list()
        
        # Merge tokens in one list
        for kb, vb in va.items():
            lst.extend(vb)
            
        # Merge in main dictionary
        main.update({ka: lst})
        
    return main

# Lemmatise tokens
def lemma(dct): 
    # Initialise lemmatizer
    wnl = nltk.WordNetLemmatizer()
    
    # Initialise an empty list
    main = dict()
    
    # Loop thru all files
    for k, v in dct.items(): 
        token_lemma = [wnl.lemmatize(t) for t in v]
        main.update({k: token_lemma})
        
    return main


if __name__ == "__main__": 
    # Set working directory
    
    # Toggle between Macbook and Windows PC
    # Macbook
    if False: 
        root = r"/Users/ongwy/Library/CloudStorage/OneDrive-Personal/Courses/NUS-ISS/MTech EBAC/Capstone"
        rptPath = r"/Users/ongwy/Library/CloudStorage/OneDrive-Personal/Courses/NUS-ISS/MTech EBAC/Capstone/ssicsync/Webscrap/AR Report Data"
    
    # PC desktop
    if True: 
        root = r'C:\Users\Wee Yang\OneDrive\Courses\NUS-ISS\MTech EBAC\Capstone'
        rptPath = r'C:\Users\Wee Yang\OneDrive\Courses\NUS-ISS\MTech EBAC\Capstone\Git\ssicsync\input_rawPDFReports'
    
    # Search all files in folder
    fileList = list(os.walk(rptPath))[0][2]
    
    # Log time
    start = datetime.now()
    
    # Extracting text from PDF 
    if True: 
        # Ingest PDF files
        print('Ingesting files...')
        print('Start time: {}'.format(str(start)))
        extracted, websites = PDFWrangling(rptPath, fileList)
        
        # Display time taken for ingestion
        print('Complete. Time taken for ingestion: {}'.format(str(datetime.now() - start)))
        
    #Saves ingested text as pickle
    if True: 
        with open(os.path.join(root, 'extractedPa.pkl'), 'wb') as f:
            pickle.dump(extracted, f)
            
        with open(os.path.join(root, 'extractedWebsite.pkl'), 'wb') as f:
            pickle.dump(websites, f)
    
    # Loads saved pickle
    if False:
        with open(os.path.join(root, 'extractedPa.pkl'), 'rb') as f:
            extracted = pickle.load(f)
            
        with open(os.path.join(root, 'extractedWebsite.pkl'), 'rb') as f:
            websites = pickle.load(f)
    
    # Log time
    start = datetime.now()
    
    print('Merging ingested pages...')
    print('Start time: {}'.format(str(start)))
    ingested = mergePages(extracted)
    
    # Display time taken for merging
    print('Time taken for merging: {}'.format(str(datetime.now() - start)))
    
    # Lemmatise tokens
    ingested = lemma(ingested)