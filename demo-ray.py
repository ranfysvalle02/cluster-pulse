import ray
import ollama
import pymongo
from pymongo import MongoClient
import os
import time
import logging
from functools import partial
from tenacity import Retrying, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError

# ==================== Configuration ====================

# Desired model for Ollama
DESIRED_MODEL = 'llama3.2'

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "")
DATABASE_NAME = "sample_mflix"
COLLECTION_NAME = "movies"

# Ray Configuration
NUM_CPUS = 4  # Adjust based on your machine's capabilities

# Batch Processing Configuration
BATCH_SIZE = 20
TOTAL_DOCUMENTS_LIMIT = 1000
CONCURRENCY = 4  # Number of batches to process concurrently

# Retry Configuration for Tenacity
RETRY_ATTEMPTS = 5
WAIT_MULTIPLIER = 1  # Initial wait time in seconds
WAIT_MAX = 10        # Maximum wait time in seconds

# Logging Configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'processing.log'

# =======================================================

# Initialize Logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Initialize Ray with specified resources
ray.init(num_cpus=NUM_CPUS)
logger.info(f"Ray initialized with {NUM_CPUS} CPUs.")

def parse_json_to_text(data):
    """
    Parses a list of JSON documents to a single formatted text string.

    Args:
        data (list): List of documents containing movie titles.

    Returns:
        str: Formatted string with titles.
    """
    texts = []
    for doc in data:
        title = doc.get('title', 'N/A')
        text = f"Title: {title}\n-----\n"
        texts.append(text)
    return "\n".join(texts)

@ray.remote
def process_batch(batch, batch_size, desired_model):
    """
    Processes a batch of documents by sending them to the Ollama model and retrieving movie titles.

    Args:
        batch (list): A subset of documents to process.
        batch_size (int): Number of documents in the batch.
        desired_model (str): The model to use for processing.

    Returns:
        list: List of extracted movie titles.
    """
    context = parse_json_to_text(batch)
    
    # Prepare the prompt for the model
    prompt = (
        "Given the [context]\n\n"
        f"[context]\n{context}\n"
        "\n[/context]\n\n"
        "RESPOND WITH A `LIST OF THE MOVIE TITLES IN THE [context];`"
        "LIST OF TITLES ONLY!, SEPARATED BY `\n` and double quotes! ESCAPE QUOTES WHEN NEEDED!"
        "YOU MUST ONLY USE [context] TO FORMULATE YOUR RESPONSE! MAKE SURE YOU RESPOND IN THE CORRECT FORMAT OR I WONT BE ABLE TO UNDERSTAND YOUR RESPONSE!"
    )
    
    messages = [
        {
            'role': 'system',
            'content': """
You will receive some [context], and your task is to respond with a list of movie titles in the [context] separated by `\n` and double quotes.
It is very important that you respond in the correct format, and never wrap your response in ``` or `.
You must only use [context] to formulate your response. Make sure you respond in the correct format or I won't be able to understand your response.
IMPORTANT! ALWAYS SEPARATE USING \n!
[response_format]
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
[/response_format]

- Must respond in the correct format, and never wrap your response in ``` or `.
- You must only use [context] to formulate your response.
- Make sure you respond in the correct format or I won't be able to understand your response.
- ALWAYS SEPARATE USING \n! A SINGLE NEWLINE!

THINK STEP BY STEP.
"""
        },
        {
            'role': 'system',
            'content': "Extract movie titles as specified."
        },
        {
            'role': 'user',
            'content': prompt,
        },
        {
            'role': 'user',
            'content': f"""
[response_format]
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
[/response_format]

IMPORTANT! ONLY RESPOND IN THIS FORMAT!
----------------------------------------------------
THINK CRITICALLY AND STEP BY STEP. EXPECTED LIST SIZE: {batch_size}
IMPORTANT! ALWAYS SEPARATE USING newline \n! NEVER SEPARATE USING SPACES, COMMAS, OR ANYTHING ELSE.
            """
        },
    ]
    
    try:
        # Implement retry logic using Tenacity's Retrying class
        retrying = Retrying(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=WAIT_MULTIPLIER, max=WAIT_MAX),
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            reraise=True
        )
        
        # Attempt to send the request with retries
        response = None
        for attempt in retrying:
            with attempt:
                response = ollama.chat(model=desired_model, messages=messages)
                logger.debug(f"Ollama response: {response}")
        
        # Initialize list to store results
        result = []
        
        # Check if the response contains content
        if response and response.get('message') and response['message'].get('content'):
            csv_output = response['message']['content']
            csv_lines = csv_output.split('\n')
            
            # Process each line to extract titles
            for line in csv_lines:
                line = line.strip()
                # Skip empty lines and unwanted characters
                if not line or line in {'```', '.', '`'}:
                    continue
                result.append(line)
        else:
            logger.warning("No response received from the model for a batch.")
        
        return result
    
    except RetryError as re:
        logger.error(f"All retry attempts failed for a batch: {re}")
        return []
    
    except Exception as e:
        logger.error(f"Error processing a batch: {e}")
        return []

def main():
    """
    Main function to orchestrate the fetching, processing, and storing of movie titles.
    """
    try:
        logger.info("Connecting to MongoDB...")
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        logger.info("Connected to MongoDB successfully.")
        
        # Fetch and format movie titles from MongoDB
        logger.info(f"Fetching up to {TOTAL_DOCUMENTS_LIMIT} movie titles from MongoDB.")
        formatted_text = list(collection.aggregate([
            {"$match": {}},
            {"$project": {"title": 1, "_id": 0}},
            {"$limit": TOTAL_DOCUMENTS_LIMIT}
        ]))
        logger.info(f"Fetched {len(formatted_text)} documents from MongoDB.")
        
        # Split the data into batches
        batches = [formatted_text[i:i + BATCH_SIZE] for i in range(0, len(formatted_text), BATCH_SIZE)]
        total_batches = len(batches)
        logger.info(f"Total batches to process: {total_batches}")
        
        final_result = []  # To store all extracted movie titles
        total_docs = 0      # Counter for processed documents
        
        # Process batches in chunks based on the concurrency limit
        for i in range(0, total_batches, CONCURRENCY):
            current_batch = batches[i:i + CONCURRENCY]
            
            logger.debug(f"Dispatching batches {i + 1} to {i + len(current_batch)}.")
            # Dispatch remote tasks to Ray
            futures = [process_batch.remote(batch, BATCH_SIZE, DESIRED_MODEL) for batch in current_batch]
            
            # Retrieve results from Ray
            results = ray.get(futures)
            
            # Collect results and update counters
            for result in results:
                final_result.extend(result)
                total_docs += len(result)
                logger.info(f"Processed a batch. Current total documents: {total_docs}")
            
            logger.info(f"Processed batches {i + 1} to {i + len(current_batch)} out of {total_batches}")
            
            # Optional: Sleep between batches to respect API rate limits
            time.sleep(1)  # Adjust or remove as needed
        
        logger.info("Processing complete.")
        logger.info(f"Total documents processed: {total_docs}/{TOTAL_DOCUMENTS_LIMIT}")
        
        # Optionally, you can handle the final_result as needed, e.g., store it or perform further processing.
    
    except Exception as e:
        logger.exception(f"An error occurred in main: {e}")
    
    finally:
        # Ensure Ray is properly shut down
        ray.shutdown()
        logger.info("Ray has been shut down.")

if __name__ == "__main__":
    main()

"""
2024-11-24 ,913	INFO worker.py:1777 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265 
2024-11-24 ,239 - INFO - Ray initialized with 4 CPUs.
2024-11-24 ,239 - INFO - Connecting to MongoDB...
2024-11-24 ,415 - INFO - Connected to MongoDB successfully.
2024-11-24 ,415 - INFO - Fetching up to 1000 movie titles from MongoDB.
2024-11-24 ,509 - INFO - Fetched 1000 documents from MongoDB.
.....................
2024-11-24 ,859 - INFO - Processed batches 41 to 44 out of 50
2024-11-24 ,145 - INFO - Processed a batch. Current total documents: 884
2024-11-24 ,145 - INFO - Processed a batch. Current total documents: 904
2024-11-24 ,145 - INFO - Processed a batch. Current total documents: 924
2024-11-24 ,145 - INFO - Processed a batch. Current total documents: 944
2024-11-24 ,145 - INFO - Processed batches 45 to 48 out of 50
2024-11-24 ,399 - INFO - Processed a batch. Current total documents: 964
2024-11-24 ,399 - INFO - Processed a batch. Current total documents: 983
2024-11-24 ,399 - INFO - Processed batches 49 to 50 out of 50
2024-11-24 ,404 - INFO - Processing complete.
2024-11-24 ,405 - INFO - Total documents processed: 983/1000
2024-11-24 ,924 - INFO - Ray has been shut down.
"""
