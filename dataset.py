import datasets
import transformers
from torch.utils.data import Dataset

token_counts_src = []
token_counts_tgt = []

def _is_valid(sentence_a, sentence_b):
    """Check if a sentence pair is valid."""
    sentence = sentence_a + ' ' + sentence_b

    '''check sentence lenght'''
    if not 5 <= len(sentence_a) <= 64 or not 5 <= len(sentence_b) <= 64:
        return False
    
    '''check if sentence contains invalid characters'''
    whitelist = '`"abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\|_+*¥"`' 
    if not all(c in whitelist for c in sentence):
        return False
    
    '''check if url in sentence'''
    if 'www.' in sentence or 'http' in sentence:
        return False
    
    '''check ratio of digits to letters'''
    if not 0.2 <= len(sentence_a) / len(sentence) <= 0.8:
        return False
        
    return True


def _map_to_task(tokenizer, example, src_bos=True):
    """Tokenize a task."""
    # TODO: use good tokenizer, add special characters

    de_en = [(x['de'],x['en']) for x in example]
    de, en = zip(*de_en)

    max_len = tokenizer.model_max_length
    src_input = tokenizer(de, truncation=True, padding='max_length', max_length=max_len+1, return_tensors='pt')['input_ids']
    if src_bos:
        src_input = src_input[:,:-1]
        assert src_input[0,0] == tokenizer.convert_tokens_to_ids('[BOS]')
    else:
        src_input = src_input[:,1:]
        assert src_input[0,0] != tokenizer.convert_tokens_to_ids('[BOS]')
    tgt = tokenizer(en, truncation=True, padding='max_length', max_length=max_len+1, return_tensors='pt')['input_ids']
    tgt_input = tgt[:,:-1] # first 512 tokens
    tgt_output = tgt[:,1:] # shift by one
    
    token_counts_src.extend(src_input.count_nonzero(dim=1).tolist())
    token_counts_tgt.extend(tgt_input.count_nonzero(dim=1).tolist())

    return {'src_input': src_input, 'tgt_input': tgt_input, 'tgt_output': tgt_output}
    

def get_costum_dataset(split='train', tokenize=True, src_bos=True):
    dataset = datasets.load_dataset("wmt17", "de-en", split=split, streaming=False)

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("modelling/bpe_v=30016_l=64")
    
    dataset = dataset.filter(lambda example: _is_valid(example['translation']['de'], example['translation']['en']))
    if tokenize:
        dataset = dataset.map(lambda example: _map_to_task(tokenizer, example['translation'], src_bos), batched=True)
    
    # delete translations
    #dataset = dataset.remove_columns('translation')

    return dataset


    
class CustomDataset():
    def __init__(self):
        self.dataset = self.dataset.filter(lambda example: _is_valid(example['translation']['de'], example['translation']['en']))
        print(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        return {
            'de': [example['translation']['de'] for example in batch],
            'en': [example['translation']['en'] for example in batch]
        }
    



if __name__ == '__main__':
    dataset = get_costum_dataset('test')
