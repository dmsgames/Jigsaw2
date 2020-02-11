from configs import Args
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm, tqdm_notebook
import numpy as np 
from torch import nn

device = torch.device('cuda')

def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            # tokens_a = tokens_a[:max_seq_length]
            half_seq_length = (int)(max_seq_length/2)
            tokens_a = tokens_a[:half_seq_length] + tokens_a[len(tokens_a)-half_seq_length:]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

class LenMatchBatchSampler(torch.utils.data.BatchSampler):

    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tsrs):
    max_len = torch.max(torch.sum( (tsrs[0] != 0  ), 1))
    if max_len > 2: 
        tsrs = [tsr[:, :max_len] for tsr in tsrs]
    return tsrs

def weighted_bce_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,0:1])(data[:,0:1],targets[:,1:2])
    return (bce_loss_1 * loss_weight) + bce_loss_2

def main():
    tokenizer = BertTokenizer.from_pretrained(Args.bert_base_uncased, do_lower_case=True)
    # read data
    if Args.debug:
        train_df = pd.read_csv(Args.train_data, nrows=50)
    else:
        train_df = pd.read_csv(Args.train_data)

    train_df['comment_text'] = train_df['comment_text'].astype(str)
    sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"), Args.max_sequence_length, tokenizer)
    train_df = train_df.fillna(0)

    X = sequences
    # subgourp
    coll = ['black','white','homosexual_gay_or_lesbian','muslim']
    weights = np.ones((len(X),)) / 4
    # Subgroup  identity_columns  > 0.5
    identity_columns = Args.identity_columns
    weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) + (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) + (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) +(train_df[coll].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) /8
    weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) +(train_df[coll].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 8
    loss_weight = 1.0 / weights.mean()
    weights = weights.reshape(-1,1)

    # y_columns = ['target']
    train_df = train_df.drop(['comment_text'], axis=1)

    y = train_df[Args.aux_columns].values
    y = np.hstack([weights,y])
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float))
    ran_sampler = torch.utils.data.RandomSampler(train_dataset)
    len_sampler = LenMatchBatchSampler(ran_sampler, batch_size = Args.batch_size, drop_last = False)

    # load trained bert model
    accumulation_steps = 2

    model = BertForSequenceClassification.from_pretrained(Args.bert_base_uncased, cache_dir=None, num_labels=len(Args.aux_columns))
    model.zero_grad()
    model = model.to(device)
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    EPOCHS = 2
    train = train_dataset
    num_train_optimization_steps = int(EPOCHS * len(train) / Args.batch_size / accumulation_steps)

    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=Args.lr)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model = model.train()

    tq = tqdm_notebook(range(EPOCHS))
    for epoch in tq:
        train_loader = torch.utils.data.DataLoader(train, batch_sampler=len_sampler)
        avg_loss = 0.
        avg_accuracy = 0.
        lossf = None
        tk0 = tqdm_notebook(enumerate(train_loader), total=len(train_loader), leave=False)
        optimizer.zero_grad()
        if epoch == 1 :
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
                param_group['warmup'] = 0
#       elif epoch == 2:
#           for param_group in optimizer.param_groups:
#               param_group['lr'] = 1e-5
#               param_group['warmup'] = 0
        for i, batch in tk0:
            tsrs = trim_tensors(batch)
            x_batch, y_batch = tuple(t.to(device) for t in tsrs)
            y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
            loss = weighted_bce_loss(y_pred,y_batch.to(device))
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98 * lossf + 0.02 * loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss=lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(
                ((torch.sigmoid(y_pred[:, 0]) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float)).item() / len(
                train_loader)
        tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)

    torch.save(model.state_dict(), Args.trained_uncased_bert)
    print('costing:%.4f S' % (time.time() - start_time))


if __name__ == "__main__":
    main()