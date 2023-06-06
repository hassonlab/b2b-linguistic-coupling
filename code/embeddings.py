'''Add embeddings to an events file that has words.

GLoVE and GPT-2 embeddings are currently supported. Though theoretically any
static model from gensim and any causal model from transformers should work.
'''

import utils
import torch
import numpy as np
import pandas as pd
import gensim.downloader as api

from tqdm import tqdm
from util.path import DerivativeBIDSPath
from transformers import AutoTokenizer #, AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from accelerate import Accelerator, find_executable_batch_size

# short names for long model names
EMBEDDINGS = {
        'glove50': 'glove-wiki-gigaword-50',
        'glove-300': 'glove-wiki-gigaword-300',
        'bbots': 'facebook/blenderbot_small-90M',
        'llama-7b': 'models/llama/7b',
        'bert-large-wwm': 'bert-large-cased-whole-word-masking'
}

random_w2v = {}

def add_static_embeddings(df, model, key='word', lowercase=True, norm=True, **kwargs):
    # To see what models gensim has:
    # info = api.info()
    # info['models']
    model = api.load(model)

    words = df[key]
    if lowercase:
        words = words.str.lower()

    def get_vector(word):
        if word in model.key_to_index:
            return model.get_vector(word, norm=norm).astype(np.float32)
        return None

    df['embedding'] = words.apply(get_vector)

    nans = df.word[df.embedding.isna()].str.lower().value_counts()
    print(f"{nans.sum()} words don't have embeddings")
    # print(nans.index.tolist())

    df.drop('utterance', axis=1, inplace=True)
    df = df.astype({'start': 'float32', 'end': 'float32', 'speaker_id': 'int32', 'utterance_id': 'int32'}, copy=False)

    return df


def add_random_embeddings(df, model, key='word', lowercase=True, seed=42, **kwargs):

    words = df[key]
    if lowercase:
        words = words.str.lower()

    ndim = kwargs.get('ndim', 1600)

    np.random.seed(seed)
    if model == 'arbitrary':
        vocab = sorted(set(words.tolist()))
        for w in vocab:
            if w not in random_w2v:
                random_w2v[w] = np.random.randn(ndim)
        # w2v = {w: np.random.randn(ndim) for w in vocab}
        df['embedding'] = words.apply(random_w2v.get)
    elif model == 'random':
        df['embedding'] = words.apply(lambda _: np.random.randn(ndim))

    return df


def add_mlm_embs(df, hfmodelname, layer=-1, device='cpu', **kwargs):
    # First, download or load model from cache
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname)
    model = AutoModelForMaskedLM.from_pretrained(hfmodelname)
    model = model.eval()

    # Tokenize input
    df.insert(0, 'word_idx', df.index.values)
    df['token'] = df.word_punc.apply(tokenizer.tokenize)
    df = df.explode('token', ignore_index=True)
    df['token_id'] = df.token.apply(tokenizer.convert_tokens_to_ids)
    tokenids = df.token_id.tolist()

    examples = []
    sentlens = []
    maxlen = tokenizer.model_max_length
    for i in df.utterance_id.unique():
        subdf = df[df.utterance_id == i]
        stop = subdf.index[-1] + 1
        start = max(0, stop - (maxlen - 2))
        sentlens.append(stop - subdf.index[0] + 1)  # add 1 for CLS
        example = [tokenizer.cls_token_id] + tokenids[start:stop] + [tokenizer.sep_token_id]
        examples.append(torch.tensor(example, dtype=torch.long))

    accelerator = Accelerator()
    model = model.to(device)

    # Run through model
    # this is NOT currently implemented to run batches
    def inference_loop(batch_size: int, layer: int):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references

        embeddings = []
        with torch.no_grad():
            data_dl = torch.utils.data.DataLoader(examples, batch_size=batch_size,
                                                  shuffle=False)
            for i, batch in enumerate(tqdm(data_dl)):
                output = model(batch.to(device), output_hidden_states=True)
                states = output.hidden_states[layer]

                batch_embeddings = states[:, -sentlens[i]:-1, :].squeeze()
                embeddings.append(batch_embeddings.numpy(force=True))

            return embeddings

    embeddings = inference_loop(batch_size=1, layer=layer)
    df['embedding'] = [e for e in np.vstack(embeddings)]
    df.drop('utterance', axis=1, inplace=True)

    return df


def add_sqs_dlm_embs(df, model, layer=-1, device='cpu'):

    # First, download or load model from cache
    tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)

    # Tokenizer input
    df.loc[df.word == '{inaudible}', 'word'] = None  # tokenizer.unk_token
    df.dropna(subset=['word'], inplace=True)
    df.insert(0, 'word_idx', df.index.values)
    df['token'] = df.word_punc.apply(tokenizer.tokenize)  # NOTE punctuation
    df = df.explode('token', ignore_index=True)
    df['token_id'] = df.token.apply(tokenizer.convert_tokens_to_ids)

    # Generate input to give model
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    maxlen = tokenizer.model_max_length
    utts = [[eos]]
    utts += [[bos] + row.token_id.values.tolist() + [eos] for _, row in
             df.groupby('utterance_id')]
    lens = [len(u) for u in utts]
    examples = []
    # examples = [{'input_ids': [eos], 'decoder_input_ids': [bos] + utts[0]}]
    for i in range(1, len(utts)):
        goback = np.sum(np.cumsum(lens[i-1::-1]) < maxlen)
        context = sum(utts[i-goback:i], [])
        examples.append({
            'input_ids': context,
            'decoder_input_ids': utts[i]
            })

    assert all([len(e['input_ids']) <= maxlen for e in examples])
    assert all([len(e['decoder_input_ids']) <= maxlen for e in examples])

    # Model
    model = AutoModelForSeq2SeqLM.from_pretrained(model, output_hidden_states=True)
    model.eval()
    model.to(device)

    df['top_pred'] = None

    embeddings = []
    with torch.no_grad():
        utts_ids = [e.index.tolist() for _, e in df.groupby('utterance_id')]
        for ids, example in zip(utts_ids, examples):
            input_ids = torch.LongTensor([example['input_ids']], device=device)
            dec_ids = torch.LongTensor([example['decoder_input_ids']], device=device)
            output = model(input_ids, decoder_input_ids=dec_ids)
            logits = output.logits[0, :-2, :]  # skip prediction for special token

            # Get top prediction
            logit_order = logits.argsort(descending=True, dim=-1)
            df.loc[ids, 'top_pred'] = logit_order[:, 0].tolist()

            dec_states = output.decoder_hidden_states[layer][0, :-2, :]
            embeddings.append(dec_states)

            # enc_states = output.encoder_hidden_states[layer][0, :, :]

    df['embedding'] = torch.cat(embeddings).tolist()
    df['top_pred'] = df.top_pred.apply(tokenizer.convert_ids_to_tokens)

    return df


def add_causal_lm_embs(df, hfmodelname, layer=-1, device='cpu', **kwargs):

    # First, download or load model from cache
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(hfmodelname)

    # random = True
    # if random:
    #     from transformers import set_seed
    #     set_seed(42)
    #     config = AutoConfig.from_pretrained(hfmodelname)
    #     model = AutoModelForCausalLM.from_config(config)  # random weights

    model = model.eval()
    maxlen = kwargs.get('maxlen', tokenizer.model_max_length)
    if maxlen is None:
        maxlen = tokenizer.model_max_length

    # Tokenize input
    df.insert(0, 'word_idx', df.index.values)
    df['token'] = df.word_punc.apply(tokenizer.tokenize)
    df = df.explode('token', ignore_index=True)
    df['token_id'] = df.token.apply(tokenizer.convert_tokens_to_ids)
    tokenids = df.token_id.tolist()

    # Do a static lookup
    if maxlen == 0:
        print('Static embeddings!')
        t2e = model.lm_head.weight
        ids = torch.tensor(df.token_id.values, dtype=torch.long)
        with torch.no_grad():
            embeddings = t2e[ids].numpy(force=True)
        df['embedding'] = [e for e in np.vstack(embeddings)]
        df = df.astype({'word_idx': 'int32', 'start': 'float32', 'end': 'float32',
                        'speaker_id': 'int32', 'utterance_id': 'int32', 'token_id': 'int32'}, copy=False)
        df.drop('utterance', axis=1, inplace=True)
        return df

    examples = []
    tokenids = torch.tensor(df.token_id.tolist(), dtype=torch.long)
    examples.append(tokenids[0:maxlen])
    for i in range(maxlen+1, len(tokenids)+1):
        examples.append(tokenids[i-maxlen:i])

    accelerator = Accelerator()
    model = model.to(device)

    # Run through model
    @find_executable_batch_size(starting_batch_size=16)
    def inference_loop(batch_size=16):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        # accelerator.print(f"Trying batch size: {batch_size}")

        top_guesses = []
        ranks = []
        true_probs = []
        entropies = []
        embeddings = []
        with torch.no_grad():
            data_dl = torch.utils.data.DataLoader(examples, batch_size=batch_size,
                                                  shuffle=False)
            # for i, batch in enumerate(tqdm(tokenids)):
            for i, batch in enumerate(tqdm(data_dl)):
                output = model(batch.to(device), output_hidden_states=True)
                logits = output.logits  # torch.Size([2, 1024, 50257])
                states = output.hidden_states[layer]

                # first case
                if i == 0:
                    true_ids = batch[0, :]
                    brange = list(range(len(true_ids)-1))
                    logits_order = logits[0].argsort(descending=True, dim=-1)
                    batch_top_guesses = logits_order[:-1, 0]
                    batch_ranks = torch.eq(logits_order[:-1], true_ids.reshape(-1,1)[1:].to(device)).nonzero()[:, 1]
                    batch_probs = logits[0, :-1].softmax(-1)
                    batch_true_probs = batch_probs[brange, true_ids[1:]]
                    batch_entropy = torch.distributions.Categorical(probs=batch_probs).entropy()
                    batch_embeddings = states[0]

                    top_guesses.append(batch_top_guesses.numpy(force=True))
                    ranks.append(batch_ranks.numpy(force=True))
                    true_probs.append(batch_true_probs.numpy(force=True))
                    entropies.append(batch_entropy.numpy(force=True))
                    embeddings.append(batch_embeddings.numpy(force=True))

                    # reset if there are more in this batch
                    if batch.size(0) == 1:
                        continue
                    logits = logits[1:]
                    states = states[1:]
                    batch = batch[1:]

                # general case
                true_ids = batch[:, -1]
                brange = list(range(len(true_ids)))
                logits_order = logits[:, -2, :].argsort(descending=True)  # batch x vocab_size
                batch_top_guesses = logits_order[:, 0]
                batch_ranks = torch.eq(logits_order, true_ids.reshape(-1,1).to(device)).nonzero()[:, 1]
                batch_probs = torch.softmax(logits[:, -2, :], dim=-1)
                batch_true_probs = batch_probs[brange, true_ids]
                batch_entropy = torch.distributions.Categorical(probs=batch_probs).entropy()
                batch_embeddings = states[:, -1, :]

                top_guesses.append(batch_top_guesses.numpy(force=True))
                ranks.append(batch_ranks.numpy(force=True))
                true_probs.append(batch_true_probs.numpy(force=True))
                entropies.append(batch_entropy.numpy(force=True))
                embeddings.append(batch_embeddings.numpy(force=True))

            return top_guesses, ranks, true_probs, entropies, embeddings

    top_guesses, ranks, true_probs, entropies, embeddings = inference_loop()

    # anything with logits should be shifted by 1
    df.loc[1:, 'rank'] = np.concatenate(ranks)
    df.loc[1:, 'true_prob'] = np.concatenate(true_probs)
    df.loc[1:, 'top_pred'] = np.concatenate(top_guesses)
    df.loc[1:, 'entropy'] = np.concatenate(entropies)
    df['embedding'] = [e for e in np.vstack(embeddings)]

    # Reduce size
    df.loc[0, 'top_pred'] = tokenizer.bos_token_id
    df = df.astype({'word_idx': 'int32', 'start': 'float32', 'end': 'float32',
                    'speaker_id': 'int32', 'utterance_id': 'int32', 'token_id': 'int32',
                    'rank': 'float32', 'true_prob': 'float32', 'top_pred': 'int32', 'entropy': 'float32'}, copy=False)
    df['top_pred'] = df.top_pred.apply(tokenizer.convert_ids_to_tokens)  # convert

    # drop unnecessary columns
    df.drop('utterance', axis=1, inplace=True)

    print('Accuracy', (df['rank'] == 0).mean())
    print('Accuracy', (df.groupby('word_idx')['rank'].first() == 0).mean())

    return df


def add_embeddings(df, model, **kwargs):
    if 'glove' in model:
        return add_static_embeddings(df, model, **kwargs)
    elif 'gpt2' in model or 'llama' in model:
        return add_causal_lm_embs(df, model, **kwargs)
    elif 'blenderbot' in model:
        return add_sqs_dlm_embs(df, model, **kwargs)
    elif model in ['arbitrary', 'random']:
        return add_random_embeddings(df, model, **kwargs)
    elif 'bert' in model:
        return add_mlm_embs(df, model, **kwargs)
    else:
        raise ValueError(model)


def main(args):
    path = DerivativeBIDSPath(subject=args.subject,
                    task=args.task,
                    root=args.root,
                    check=False)

    modelname = args.modelname
    suffix = ''
    if args.maxlen is not None:
        suffix += f'_maxlen-{args.maxlen}'
    if args.layer is not None:
        suffix += f'_layer-{args.layer}'
    if args.ndim is not None:
        suffix += f'_ndim-{args.ndim}'
    if modelname in ['arbitrary', 'random']:
        suffix += f'_seed-{args.seed}'

    # random = True
    # if random:
    #     suffix += '_random'
    suffix += '_ztest'

    outpath = path.copy()
    outpath.update(root=f'{args.root}/derivatives/embeddings', extension='.pkl')
    outpath.update(datatype=modelname, suffix=f'model-{modelname}{suffix}_embeddings')
    print(outpath.fpath)

    # Get embeddings
    df = utils.gettranscript(path)
    df = add_embeddings(df, **vars(args))

    # Write out
    outpath.mkdir(exist_ok=True)
    df.to_pickle(outpath.fpath)


def manual_files(args):
    df = pd.read_csv(args.infile, index_col=0)
    for emb in args.embeddings:
        name = EMBEDDINGS.get(emb, emb)
        df = add_embeddings(df, name, device=args.device)
        outfn = args.outfile.replace('.csv', f'_desc-{emb}_embeddings.csv')
        df.to_pickle(outfn)


def copy_to_partners(args):
    emb = args.model
    hfmodelname = EMBEDDINGS.get(args.model, args.model)  # get long name
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname, add_prefix_space=True)

    suffix = ''
    if args.maxlen is not None:
        suffix += f'_maxlen-{args.maxlen}'
    if args.layer is not None:
        suffix += f'_layer-{args.layer}'

    # random = True
    # if random:
    #     suffix += '_random'

    for sub in args.subject:
        # get transcript for partner

        part = utils.getpartner(sub)
        path = DerivativeBIDSPath(subject=f'{part:02d}', task=args.task, root=args.root)
        df = utils.gettranscript(path)

        # get embeddings for sub
        outpath = path.copy()
        outpath.update(subject=f'{sub:02d}', root=f'{args.root}/derivatives/embeddings', extension='.pkl')
        outpath.update(datatype=emb, suffix=f'model-{emb}{suffix}_embeddings')
        dfemb = pd.read_pickle(outpath.fpath)

        # explode into tokens
        df.insert(0, 'word_idx', df.index.values)
        df['token'] = df.word_punc.apply(tokenizer.tokenize)
        df = df.explode('token', ignore_index=True)
        df['token_id'] = df.token.apply(tokenizer.convert_tokens_to_ids)

        # ensure they're perfectly aligned (only diff should be onset/offset)
        assert len(df) == len(dfemb), 'dyad transcripts must be same size and aligned'

        # Copy data into it
        for col in ['rank', 'true_prob', 'top_pred', 'entropy', 'embedding']:
            if col in dfemb.columns:
                df[col] = dfemb[col]

        # Write out
        outpath.update(subject=f'{part:02d}')
        outpath.mkdir(exist_ok=True)
        print(outpath.fpath)
        df.to_pickle(outpath.fpath)


if __name__ == '__main__':
    parser = utils.getparser()
    parser.add_argument('infile', nargs='?')
    parser.add_argument('outfile', nargs='?')
    parser.add_argument('-m', '--model', default='glove50')
    parser.add_argument('--maxlen', type=int, default=None)
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--ndim', type=int, default=None)  # for arbitrary and random
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()

    if args.copy:
        copy_to_partners(args)
        exit()

    args.device = torch.device('cpu')
    if torch.cuda.is_available() and not args.force_cpu:
        args.device = torch.device('cuda', args.cuda)
    else:
        print('WARNING: using cpu only')

    if args.infile is not None and args.outfile is not None:
        manual_files(args)
        exit()

    args.modelname = args.model
    args.model = EMBEDDINGS.get(args.model, args.model)  # get long name

    subjects = args.subject
    for subject in subjects:
        args.subject = f'{subject:02d}'
        main(args)