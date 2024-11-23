import ollama
desiredModel = 'llama3.2'

# demo setup
def parse_json_to_text(data):
    print("JSON data length:", len(data))
    print("Parsing JSON to text...")
    texts = []
    for doc in data:
        title = doc.get('title', 'N/A')
        
        text = (
            f"Title: {title}\n"
            "-----\n"
        )
        texts.append(text)
    
    return "\n".join(texts)

# Example usage:
if __name__ == "__main__":
    try:
        # Load text from MongoDB
        import pymongo
        client = pymongo.MongoClient("mongodb+srv://__DEMO__")
        db = client["sample_mflix"]
        collection = db["movies"]
        formatted_text = list(collection.aggregate([
            {"$match": {}},
            {"$project": {"title": 1}},
            {"$limit": 1000}
        ]))
        # make sure we avoid `An error occurred: Object of type ObjectId is not JSON serializable`
        # turn it into a JSON string I can pass to parse_json_to_text
        formatted_text = parse_json_to_text((formatted_text))

        context = formatted_text
        
        # Prepare the prompt for the model
        prompt = (
            "[INST]<<SYS>>RESPOND WITH A `COMPLETE LIST OF THE 1000 MOVIE TITLES` IN THE [context]\n\n"
            f"[context]\n{context}\n"
            "\n[/context]<</SYS>> RESPOND WITH A `COMPLETE LIST OF THE 1000 MOVIE TITLES; NO OMMISSIONS! MUST BE COMPLETE LIST!`[/INST]"
        )
        
        print("Prompt:", prompt)
        # Interact with the Ollama model
        res = ollama.chat(model=desiredModel, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        # Output the response
        if res.get('message') and res['message'].get('content'):
            print(res['message']['content'])
        else:
            print("No response received from the model.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

"""
JSON data length: 1000
=======
Here is the complete list of 1000 movie titles:

1. ...And God Created Woman
2. ...But Not for Me
3. A Man Called Peter
4. A Man in a Barroom
5. A Man with the Golden Arm
..........
439. Alligator
440. Along Side You
441. The Amazing Transplant
442. Amélie Poulain
"""
