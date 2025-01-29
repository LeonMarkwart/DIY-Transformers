from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import tqdm

from dataset import get_costum_dataset
from modelling.model import Transformer
from modelling.lr_scheduler import TransformerLRScheduler
from transformers import GPT2Tokenizer
import pytorch_lightning as pl
import evaluate

from dataset import get_costum_dataset

def collate_fn(sample):
    src_input = [s['src_input'] for s in sample]
    tgt_input = [s['tgt_input'] for s in sample]
    tgt_output = [s['tgt_output'] for s in sample]
    return {
        'src_input': torch.tensor(src_input),
        'tgt_input': torch.tensor(tgt_input),
        'tgt_output': torch.tensor(tgt_output)
    }
    

class TransformerModel(pl.LightningModule):

    def __init__(self, 
                 max_lr=None, # if it is not none, the lr scheduler automatically scales up to max_lr, otherwise it uses vanilla TransformerLRScheduler
                 bs=32, 
                 label_smoothing=0.1,
                 vocab_size=30_016, # roundabout 30k, like in Attention is all you need, but dividable by 64
                 d_model=512,
                 n_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=512*4,
                 warmup_steps=4000,
                 max_len=64,
                 dropout=0.1,
                 src_BOS_token=True, # True if the src_input should have a leading BOS token
                 ):
        super().__init__()
        self.save_hyperparameters()

        model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.model = model
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'modelling/bpe_v={vocab_size}_l={max_len}')

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=0)

        """Log the number of trainable parameters."""
        self.hparams['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.bleu_metric = evaluate.load("bleu")

    def forward(self, src_input, tgt_input):
        # assert that the first input_tgt token is BOS for all samples
        assert torch.all(tgt_input[:,0] == self.tokenizer.convert_tokens_to_ids('[BOS]'))
        src_input_mask = (src_input != 0).int()
        tgt_input_mask = (tgt_input != 0).int()
        return self.model(src_input, tgt_input, src_input_mask, tgt_input_mask)
    
    def training_step(self, batch, batch_idx):
        src_input = batch['src_input']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']

        """Calculate the loss and log it."""
        output = self(src_input, tgt_input)
        loss_step = self.loss(output.view(-1, output.size(-1)), tgt_output.view(-1))
        self.log('train_loss', loss_step, prog_bar=True, on_epoch=True, on_step=True, logger=True) 

        return loss_step
    

    def validation_step(self, batch, batch_idx):
        src_input = batch['src_input']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']
        
        """Calculate the loss and log it."""
        output = self(src_input, tgt_input)
        loss_step = self.loss(output.view(-1, output.size(-1)), tgt_output.view(-1))
        self.log('val_loss', loss_step, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        """Calculate the BLEU score and log it."""
        greey_prediction = self.greedy_translate(src_input)
        self.calculate_bleu(src_input, tgt_output, greey_prediction, batch_idx, log=True)

        """Print the first sample of the batch in unit test mode."""
        if self.trainer.fast_dev_run:
            print('src_input:\ntgt_input:\ntgt_output:\ngreedy_prediction:\n')
            for i in range(src_input.size(0)):
                print(self.tokenizer.decode(src_input[i], skip_special_tokens=True))
                print(self.tokenizer.decode(tgt_input[i], skip_special_tokens=True))
                print(self.tokenizer.decode(tgt_output[i], skip_special_tokens=True))
                print(self.tokenizer.decode(greey_prediction[i], skip_special_tokens=True))
                print()

        return loss_step
    
    def test_step(self, batch, batch_idx):
        src_input = batch['src_input']
        tgt_output = batch['tgt_output']

        """Calculate the BLEU score"""
        prediction = self.greedy_translate(src_input)        
        bleu_score, text_batch = self.calculate_bleu(src_input, tgt_output, prediction, batch_idx, log=True)

        return bleu_score, text_batch


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98), eps=1e-09, weight_decay=0.01, fused=False) # lr here doesnt matter if max_lr is passed to scheduler

        scheduler = TransformerLRScheduler(optimizer, self.hparams.d_model, self.hparams.warmup_steps, max_lr=self.hparams.max_lr)

        print(f"Set Learning Rate: {self.hparams.max_lr}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
    
    def train_dataloader(self):
        train_loader = DataLoader(
            get_costum_dataset('train', src_bos=self.hparams.src_BOS_token),
            batch_size=self.hparams.bs,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            get_costum_dataset('validation', src_bos=self.hparams.src_BOS_token),
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(
            get_costum_dataset('test', src_bos=self.hparams.src_BOS_token),
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        return test_loader
    

    def greedy_translate(self, src_input):
        """
        Predicts the most likely next token and feeds it back into the model.
        Stops if all tokens are PAD tokens. (Rarely happens because model doesnt always respect EOS token)
        """
        tgt_input = torch.full_like(src_input, self.tokenizer.pad_token_id)
        tgt_input[:,0] = self.tokenizer.convert_tokens_to_ids('[BOS]')


        for i in range(1, src_input.size(1)):
            output = self(src_input, tgt_input)
            output = output.argmax(dim=-1)
            tgt_input[:,i] = output[:,i-1]
            if torch.all(tgt_input[:,i] == self.tokenizer.convert_tokens_to_ids('[PAD]')):
                break
            
        return output
    
    def calculate_bleu(self, src_input, tgt_output, prediction, batch_idx, log=False):
        """
        Calculate BLEU score for the given prediction and target output.
        Also logs the BLEU score and the first sample of the batch.
        """
        # Decode the output and target sequences
        predicted_output = [self.tokenizer.decode(o, skip_special_tokens=True) for o in prediction]
        tgt_output = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tgt_output]
        src_input = [self.tokenizer.decode(s, skip_special_tokens=True) for s in src_input]

        # Delete EOS and everthing after it
        predicted_output = [o.split('[EOS]')[0] for o in predicted_output]
        tgt_output = [t.split('[EOS]')[0] for t in tgt_output]
        src_input = [s.split('[EOS]')[0] for s in src_input]

        # Compute BLEU score
        bleu_score = self.bleu_metric.compute(predictions=predicted_output, references=tgt_output)
        if log:
            self.log('bleu', bleu_score['bleu'], prog_bar=True, on_epoch=True, on_step=False, logger=True)
            self.log('hp_metric', bleu_score['bleu'], on_epoch=True, on_step=False, logger=True)

        if log and batch_idx == 0:
            log = "| src_input | tgt_output | prediction |  \n"
            log += "|--------|--------|------------|  \n"
            for i in range(len(tgt_output)):
                log += f'| {src_input[i]} | {tgt_output[i]} | {predicted_output[i]} |  \n'
            self.logger.experiment.add_text('Sample', log, self.global_step)

        return bleu_score, {'src_input': src_input, 'tgt_output': tgt_output, 'prediction': predicted_output}
    

if __name__ == "__main__":

    model = TransformerModel.load_from_checkpoint('experiment/main/version_0/checkpoints/epoch=14-step=181515.ckpt')


    trainer = pl.Trainer(fast_dev_run=False)
    trainer.test(model)



    exit()

    model = TransformerModel(
        bs=64,
        src_BOS_token=False,
        num_encoder_layers=8,
        num_decoder_layers=8,)
    

    trainer = pl.Trainer(
        callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='step')],
        precision='16-mixed',
        gradient_clip_val=1,
        #accumulate_grad_batches=4,
        fast_dev_run=False,
        #max_epochs=10
        )

    trainer.fit(model)
