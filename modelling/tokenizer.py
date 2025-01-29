import tokenizers
import tokenizers.implementations
import sys
sys.path.append('/Users/leonmarkwart/Library/Mobile Documents/com~apple~CloudDocs/HHU/Semester 3/Implementing Transformers/implementingtransformers/')
from dataset import CustomDataset

class CustomTokenizer():
    def __init__(self):
        self.tokenizer = tokenizers.implementations.CharBPETokenizer()

    def train(self, dataset):
        # Train the tokenizer
        self.tokenizer.train_from_iterator(dataset, vocab_size=50000)
        # Save the tokenizer
        self.tokenizer.save("tokenizer.json")
        
    def __call__(self, text):
        return self.tokenizer.encode(text).ids
    
if __name__ == '__main__':
    tokenizer = CustomTokenizer()
    ds = CustomDataset('train').dataset

    sentence_list = []

    for sentence in ds['train']['translation']:
        sentence_list.append(sentence['de'])
        sentence_list.append(sentence['en'])

    tokenizer.train(sentence_list)