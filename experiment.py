import argparse

import numpy as np
import pandas as pd
import yaml, os, re
import time
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from transformers import pipeline, AutoTokenizer


def remove_name_entities(sentence):
    # Define the pattern to match "/NAMEENTITY"
    pattern = r'/NAMEENTITY'

    # Remove all occurrences of "/NAMEENTITY" from the sentence
    cleaned_sentence = re.sub(pattern, '', sentence)

    return cleaned_sentence


def extract_words(sentence):
    # Define the pattern to match word preceding "/NAMEENTITY"
    pattern = r'(\w+)/NAMEENTITY'

    # Find all matches in the sentence
    matches = re.findall(pattern, sentence)

    return matches


import re

def extract_word_after_slash(context):
    # Use regular expression to find all words after '/'
    pattern = re.compile(r'/(?P<word>\w+)')
    words = pattern.findall(context)
    return words

def compare_contexts(original_context, predicted_context):
    original_words = extract_word_after_slash(original_context)
    predicted_words = extract_word_after_slash(predicted_context)

    # Ensure both lists are of the same length by truncating the longer one
    min_length = min(len(original_words), len(predicted_words))
    original_words = original_words[:min_length]
    predicted_words = predicted_words[:min_length]

    comparison_results = []

    for o_word, p_word in zip(original_words, predicted_words):
        comparison_results.append((o_word, p_word, o_word == p_word))

    return comparison_results



def run():
    embedding_llm = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        # model_kwargs = {'device' : 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )

    loader = DirectoryLoader(
        'ragdata/',
        loader_cls=TextLoader,
        glob="./*.txt"
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=2500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_llm,
        persist_directory='db/00'
    )

    sentences = []
    with open('new.txt', "r", encoding="utf") as file:
        for i in file:
            sentences.append(i)
        len(sentences)

    list_of_tags = ["LOC", "GPE", "CAMP", "GHETTO", "STREET"]
    tags_meaning = ["LOC : Locations except country or cities ", "GPE : Geographical location like country or a city",
                    "CAMP : Concentration Camps", "GHETTO : Ghetto, the Jewish quarter in a city.",
                    "STREET : path ways or Roads "]

    retriever = vector_db.as_retriever(search_kwargs={"k": 2}, search_type="similarity")
    filefinal = open("a_predict.txt", "w", encoding='utf')

    for sentence in sentences:
        cleansen = remove_name_entities(sentence)
        words = extract_words(sentence)
        docs = retriever.get_relevant_documents(cleansen)

        result = ""
        for i in docs:
            result += i.page_content

        # if len(result.split(" ")) > 32769:
        #     result = result.split(" ")[:32769]
        #     result = " ".join(result)

        query = f'''Consider the Year from 1936-1944. You are going to identify Name entity tags are for holocaust specific tags. The list of name entity tags should be {list_of_tags}.
                    Each tag is as follows. {tags_meaning}
                    Now do the below tasks.
                    1. Examine the below examples and learn about the appropriate Name entity tags for the words based on the context. words are tagged with appropriate entity with / Examples are '{result}'
                    2. Now try to identify the most suitable Name entity tag for word 'NAMEENTITY' in the GIVEN SENTENCE based in the below criterias.
                    - Analyse the word infront of the NAMEENTITY tag before you tag.
                    - Get the understanding of the complete sentence and try to identify specific factors discuss about the word you want to tag.
                    The GIVEN SENTENCE :  {sentence}.
                    3. Return only the GIVEN SENTENCE after assigning the identified tags instead of the word 'NAMEENTITY'. Do not add additional data or completion statment.
                    
                    Use the following format for the output
                    <sentence with name entity tags>'''

        answer = pipe(
            query,
            max_new_tokens=2048,
            temperature=0,
            num_return_sequences=1,
            do_sample=True,
        )
        filefinal.write(answer)
        print(answer)

    filefinal.close()

    correct_prediction = 0
    incorrect_prediction = 0
    with open("a_predict.txt", 'r', encoding='utf-8') as predictfile, open("a_real.txt", 'r',
                                                                           encoding='utf-8') as realfile:
        # Example usage
        for original_context, predicted_context in zip(realfile, predictfile):

            comparison_results = compare_contexts(original_context, predicted_context)

            for o_word, p_word, is_same in comparison_results:
                if is_same:
                    correct_prediction += 1
                else:
                    incorrect_prediction += 1
                # print(f"Original: {o_word} - Predicted: {p_word} - Match: {is_same}")
                print(f"{o_word},{p_word}")

            # print("\n")

    print("correct_prediction", correct_prediction)
    print("incorrect_prediction", incorrect_prediction)

    original = []
    predicted = []
    with open("results.txt", "r", encoding='utf-8') as f:
        for i in f:
            i = i.strip("\n")
            original.append(i.split(",")[0])
            predicted.append(i.split(",")[1])

    print(original)
    print(predicted)
    # List of classes
    classes = ["LOC", "GPE", "CAMP", "GHETTO", "STREET"]

    # Compute confusion matrix
    cm = confusion_matrix(original, predicted, labels=classes)

    # Compute classification report
    report = classification_report(original, predicted, labels=classes, target_names=classes)

    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def get_chat_template():
    # https://github.com/chujiezheng/chat_templates/tree/main/chat_templates
    chat_template = None
    if str(args.model_name).__contains__('mistral'):
        chat_template = open('templates/mistral-instruct.jinja').read()
    elif str(args.model_name).__contains__('falcon'):
        chat_template = open('templates/falcon-instruct.jinja').read()
    elif str(args.model_name).__contains__('Llama-2') or str(args.model_name).__contains__('Saul-7B'):
        chat_template = open('templates/llama-2-chat.jinja').read()
    elif str(args.model_name).__contains__('Meta-Llama-3'):
        chat_template = open('templates/llama-3-instruct.jinja').read()
    elif str(args.model_name).__contains__('Phi-3'):
        chat_template = open('templates/phi-3.jinja').read()
    return chat_template


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''judgement prediction in UKSC cases''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    args = parser.parse_args()

    tokenizer_mt = AutoTokenizer.from_pretrained('local_models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    chat_template = get_chat_template()
    if chat_template:
        tokenizer_mt.chat_template = chat_template
    pipe = pipeline(
        "text-generation",
        model='local_models/Meta-Llama-3.1-8B-Instruct',
        model_kwargs={"torch_dtype": torch.bfloat16,
                      # "attn_implementation": "flash_attention_2" if str(args.model_name).__contains__(
                      #     'Phi-3') else None
                      },
        device_map="auto",
        tokenizer=tokenizer_mt,
        trust_remote_code=True
    )
    # pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'
    run()
