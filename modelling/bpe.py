from dataset import get_costum_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import GPT2Tokenizer
import json
from dataset import get_costum_dataset
import os
import datasets
import transformers
from tqdm import tqdm


def train_tokenizer(vocab_size=30_016, max_length=64):
    dataset = get_costum_dataset(split='train', tokenize=False)
    print(len(dataset))
    data_txt = [f"{item['translation']['de']} {item['translation']['en']}" for item in tqdm(dataset)]

    bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    bpe_tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], # no masking because already done in attention 
        vocab_size=vocab_size,
        show_progress=True,
        max_token_length=max_length
    )

    bpe_tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", 1),
            ("[EOS]", 2),
        ],
    )
    
    bpe_tokenizer.train_from_iterator(data_txt, trainer=trainer)

    gpt2 = transformers.GPT2TokenizerFast(tokenizer_object=bpe_tokenizer)
    gpt2.pad_token = '[PAD]'
    gpt2.pad_token_id = gpt2.convert_tokens_to_ids('[PAD]')
    gpt2.model_max_length = max_length

    sentence = "Hallo, mein Name ist Leon. Wie kann ich helfen?"
    encode = gpt2(sentence, truncation=True, padding='max_length', return_tensors='pt')
    decode = gpt2.decode(encode['input_ids'][0])
    assert sentence in decode

    
    
    #save tokenizer
    os.makedirs(f"./modelling/bpe_v={vocab_size}_l={max_length}", exist_ok=True)
    gpt2.save_pretrained(f"./modelling/bpe_v={vocab_size}_l={max_length}")



class CustomBPETokenizer:

    dataset = get_costum_dataset(tokenize=False)

    def __init__(self, vocab_size=10_000, max_length=256):
        data_txt = [f"{item['de']} {item['en']}" for item in self.dataset]
        self.data = data_txt
        self.bpe_tokenizer = self.train_bpe_tokenizer(vocab_size)
        self.gpt2_tokenizer = self.convert_to_gpt2_tokenizer()
        self.max_length = max_length

    def train_bpe_tokenizer(self, vocab_size):
        bpe_tokenizer = Tokenizer(models.BPE())

        bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        bpe_tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], # no masking because already done in attention 
            vocab_size=vocab_size,
            show_progress=True,
        )

        bpe_tokenizer.train_from_iterator(self.data, trainer=trainer)

        return bpe_tokenizer

    def convert_to_gpt2_tokenizer(self):
        model = json.loads(self.bpe_tokenizer.to_str())['model']
        vocab_dict = model['vocab']
        merges_list = model['merges']

        os.makedirs("./modelling/bpe_v=10000", exist_ok=True)
        with open("./modelling/bpe_v=10000/vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        with open("./modelling/bpe_v=10000/merges.txt", "w") as merges_file:
            merges_file.write("\n".join(" ".join(map(str, merge)) for merge in merges_list))

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./modelling/bpe_v=10000")
        gpt2_tokenizer.pad_token = ...

        return gpt2_tokenizer

    def tokenize(self, example):
        return self.gpt2_tokenizer.tokenize(example)

    def encode(self, example):
        return self.gpt2_tokenizer.encode(example)

    def decode(self, tokens):
        return self.gpt2_tokenizer.decode(tokens)

train_tokenizer()