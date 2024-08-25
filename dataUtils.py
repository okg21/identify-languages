import os
import numpy as np
import matplotlib.pyplot as plt

def readConnluFiles(dataset_path):
    # Dictionary to hold the contents of the .conllu files, organized by type
    conllu_contents = {'train': {}, 'dev': {}, 'test': {}}
    
    # Loop through each item in the dataset directory
    for item in os.listdir(dataset_path):
        # Construct the full path of the item
        full_path = os.path.join(dataset_path, item)
        
        # Check if the item is a directory and starts with 'UD_'
        if os.path.isdir(full_path) and item.startswith('UD_'):
            # Scan the directory for .conllu files
            for file in os.listdir(full_path):
                if file.endswith('.conllu'):
                    # Identify the type of the file based on its name
                    file_type = ''
                    if 'train' in file:
                        file_type = 'train'
                    elif 'dev' in file:
                        file_type = 'dev'
                    elif 'test' in file:
                        file_type = 'test'
                    
                    if file_type:  # Ensure file_type is identified
                        # Construct the full path to the .conllu file
                        file_path = os.path.join(full_path, file)
                        # Read the contents of the .conllu file
                        with open(file_path, 'r') as file:
                            conllu_contents[file_type][item] = file.readlines()
                            
    # Print some samples
    for file_type, languages in conllu_contents.items():
        print(f"Read files. Contents for {file_type}:")
        for language, contents in languages.items():
            print(f"{language}:")
            print(contents[:2])  
    
    return conllu_contents



def extractTexts(conllu_files):
    # Dictionary to store the extracted texts from each language's file, organized by file type
    extracted_texts = {'train': {}, 'dev': {}, 'test': {}}
    
    # Iterate over each file type and its languages
    for file_type in conllu_files.keys():
        for language, lines in conllu_files[file_type].items():
            # List to hold the extracted texts for the current language and file type
            language_texts = []
            
            # Process each line to find and extract text after 'text ='
            for line in lines:
                if 'text =' in line.strip():
                    # Extract the text following 'text ='
                    text = line.strip().split('text = ')[1].strip()
                    language_texts.append(text)
            
            # Store the extracted texts in the dictionary under the current language and file type
            if language_texts:  # only add if there are texts extracted to avoid empty entries
                extracted_texts[file_type][language] = language_texts
                
    # Print some samples
    for file_type, languages in extracted_texts.items():
        print(f"Extracted texts for {file_type}:")
        for language, texts in languages.items():
            print(f"{language}:")
            for i, text in enumerate(texts):
                if i < 5:
                    print(text)
                else:
                    break
    
    return extracted_texts


# Assuming texts_by_file_type is available from previous steps where texts are organized by 'train', 'dev', 'test'
def createLabeledDataset(texts_by_file_type):
    datasets = {}

    # Process each file type ('train', 'dev', 'test')
    for file_type, languages_texts in texts_by_file_type.items():
        data = []
        for language, texts in languages_texts.items():
            for text in texts:
                data.append((text, language))  # Create tuple (sentence, language label)

        # Convert list to a numpy array for easier manipulation
        data_array = np.array(data, dtype=object)  

        # Shuffle the data to ensure random distribution
        np.random.seed(42)
        np.random.shuffle(data_array)

        # Accessing sentences and labels
        sentences = data_array[:, 0]
        labels = data_array[:, 1]

        # Store in a dictionary
        datasets[file_type] = {'sentences': sentences, 'labels': labels}
        
    # Print some samples
    for file_type, dataset in datasets.items():
        print(f"{file_type} dataset size: {len(dataset['sentences'])} samples")
        print("Sample data:")
        for i in range(3):  
            print(f"Sentence: {dataset['sentences'][i]}, Label: {dataset['labels'][i]}")

    return datasets



def plotLabelDistributions(datasets):
    # Set up the matplotlib figure
    plt.figure(figsize=(18, 6))  # Width, Height
    
    # Labels for subplots
    dataset_labels = ['Train', 'Validation', 'Test']
    
    # Loop through the datasets dictionary
    for i, (key, data) in enumerate(datasets.items(), 1):
        # Calculate the unique labels and their frequencies
        labels, counts = np.unique(data['labels'], return_counts=True)
        
        # Create a subplot for each dataset
        plt.subplot(1, 3, i)
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f'{dataset_labels[i-1]} Dataset Label Distribution')
    
    # Show the plot
    plt.tight_layout()
    plt.show()