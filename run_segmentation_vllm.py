import argparse
import multiprocessing
from tqdm import tqdm
import pandas as pd
import os
import re
import random
import time
from openai import OpenAI

num_proc = 24

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--path', type=str, required=True, help='Path argument')

args = parser.parse_args()
path = args.path

def get_baseurl():
    # get random number between 0 and 7
    random_number = random.randint(0, 7)
    random_number = 0 
    base_url=f"http://localhost:340{str(random_number)}/v1"
    return base_url
    
def get_segmentation(text):
    # if text only contains whietspace, return empty string
    client = OpenAI(        
        base_url=get_baseurl(),
        api_key="123",
    )
    prompt = f"""
    <bos><start_of_turn>user
    Here is a list of sentences that need to be segmented into coherent paragraphs based on their content:
    {text}
    Please segment them into paragraphs. after each paragraph, insert a # character. Return the full paragraph. The ideal length is between 4 and 10 sentences. Do not explain your steps, only return the paragraphs followed by the # character. Also remove other noise such as OCR artifacts, page headers/footers, footnote numbers etc.:<end_of_turn>
    <start_of_turn>model
    """
    print(prompt)    
    attempts = 0 
    while attempts < 20:
        try:
            response = client.completions.create(
                model="google-gemma2",
                prompt=prompt,
                #stop=["#"],
                max_tokens=1000, # max tokens for input+output is 4096, so this should be 4096-input_tokens
                temperature=0.0, # less temperature = less hallucination
                stream=False # we want to get full sentences
            )
            target_text = response.choices[0].text            
            print(target_text)
            target_text = re.sub(r"[^0-9\s]", "", target_text)
            return target_text
        
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {str(e)}")
            attempts += 1
            if attempts < 20:
                print(f"Retrying in 5 seconds...")
                time.sleep(2)
            else:
                print("Max attempts reached. Giving up.")
                return ""


def preprocess_for_segmentation(sentences):
    i = 0
    result = ""
    for sentence in sentences:
        result += f"{i}. {str(sentence)[:250]}\n"
        i += 1
    return result

def test_if_sentence_has_punctuation(sentence):
    # test if sentence is str
    if not isinstance(sentence, str):
        return False
    if re.search(r"[.!?]", sentence):
        return True
    return False

def process_tsv(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            df = pd.read_csv(f, sep="\t", engine="python", on_bad_lines="skip")
            
        if df.empty:
            print(f"Skipping empty file: {path}")
            return
        
        # drop rows with empty translation
        df = df.dropna(subset=['translated'])
        
        if df.empty:
            print(f"Skipping file with no valid translations: {path}")
            return
        
        # convert translation to string 
        df['translated'] = df['translated'].astype(str)
        # add empty paragraph column
        df['paragraph'] = 0
        current_paragraph = 0
        
        # iterate over df in steps of 20 rows
        for i in range(0, len(df), 30):
            max = 30 if i + 30 < len(df) else len(df) - i
            sentences = df.iloc[i:i+max]['translated'].tolist()
            sentences = preprocess_for_segmentation(sentences)
            break_points = get_segmentation(sentences)
            # break points is a list of sentence numbers at which a new paragraph starts
            break_points = [int(bp) for bp in break_points.split()]
            break_points = [0] + break_points
            for j in range(len(sentences)):
                if i+j >= len(df):
                    break
                # test if df index has i+j
                if i+j-1 in df.index: 
                    if j in break_points and test_if_sentence_has_punctuation(df.at[i+j-1, 'translated']):
                        current_paragraph += 1
                df.at[i+j, 'paragraph'] = current_paragraph
        
        df.to_csv(path.replace(".tsv", "-segmented.tsv"), sep="\t", index=False)
        print(f"Successfully processed: {path}")
    
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {path}")
    except pd.errors.ParserError:
        print(f"Skipping file with parsing error: {path}")
    except Exception as e:
        print(f"Error processing file {path}: {str(e)}")



for dirpath, dirnames, filenames in os.walk(path):
    list_of_files = []
    for filename in filenames:
        if filename.endswith("translated.tsv") and not filename.endswith("-segmented.tsv") and "segmented" not in filename:
            full_path = os.path.join(dirpath, filename)
            translated_path = os.path.join(dirpath, filename.replace(".tsv", "-segmented.tsv"))
            if not os.path.exists(translated_path):
                print(full_path)
                list_of_files.append(full_path)

    with multiprocessing.Pool(processes=num_proc) as pool:    
        for _ in tqdm(pool.imap_unordered(process_tsv, list_of_files), total=len(list_of_files)):
            pass
    
