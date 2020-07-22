import os
import re
import codecs
from nltk import TreebankWordTokenizer, WordPunctTokenizer
from transformers import BertTokenizer


task2_labels = {
    'Appeal_to_Authority': 0,
    'Appeal_to_fear-prejudice': 1,
    'Bandwagon,Reductio_ad_hitlerum': 2,
    'Black-and-White_Fallacy': 3,
    'Causal_Oversimplification': 4,
    'Doubt': 5,
    'Exaggeration,Minimisation': 6,
    'Flag-Waving': 7,
    'Loaded_Language': 8,
    'Name_Calling,Labeling': 9,
    'Repetition': 10,
    'Slogans': 11,
    'Thought-terminating_Cliches': 12,
    'Whataboutism,Straw_Men,Red_Herring': 13,
    0: 'Appeal_to_Authority',
    1: 'Appeal_to_fear-prejudice',
    2: 'Bandwagon,Reductio_ad_hitlerum',
    3: 'Black-and-White_Fallacy',
    4: 'Causal_Oversimplification',
    5: 'Doubt',
    6: 'Exaggeration,Minimisation',
    7: 'Flag-Waving',
    8: 'Loaded_Language',
    9: 'Name_Calling,Labeling',
    10: 'Repetition',
    11: 'Slogans',
    12: 'Thought-terminating_Cliches',
    13: 'Whataboutism,Straw_Men,Red_Herring',
    '?': -1
}

def get_span_labels(file='data/labels/train-task1-SI.labels'):
    f_labels = {}
    with open(file, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        split_line = line.split('\t')
        numbers = [int(split_line[1]), int(split_line[2])]
        if split_line[0] in f_labels:
            f_labels[split_line[0]].append(tuple(numbers))
        else:
            f_labels[split_line[0]] = [tuple(numbers)]
    return f_labels


def sub_unicode_chars(text):
    text = re.sub('[’‘]', "'", text)
    text = re.sub('[”“]', '"', text)
    text = re.sub('[·]', ".", text)
    text = re.sub("[—–]", "-", text)
    text = re.sub(" '\u200f' | '\ufeff' | '\u200b' | '\u202c | u'\x9d'", " ", text)
    text = re.sub("''", '""', text)
    return text


def get_regex():
    url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    twitter_regex = r"—.*\d, \d{4}$"
    num_regex = r"[\[\(][\d,]+[\]\)]"
    at_regex = r"@\S+"
    ad_regex = r'—.*\- Details\)'
    ref_regex = r"\(.{,10}:.{,10}\)"
    source_regex = r"\(source\)"
    weird_parenthesis_regex = r"^\d+:.{,5}\.'?"
    time_regex = r"\d{2}:\d{2} ?(etc|[AaPp)\.?[Mm]\.?)"
    regex = re.compile("{}|{}|{}|{}|{}|{}|{}|{}|{}".format(url_regex, twitter_regex, at_regex, ad_regex, time_regex, weird_parenthesis_regex, source_regex, ref_regex, num_regex))
    return regex


def load_data(path='data/train-articles'):
    articles = {}
    paths = os.listdir(path)
    paths.sort()
    paths = list(map(lambda p: os.path.join(path, p), paths))
    for p in paths:
        with codecs.open(p, 'r', encoding='utf-8') as file:
            text = file.read()
        file_key = re.sub('\D+', '', p)
        articles[file_key] = text

    return articles


def get_class_labels(file='data/labels/train-task2-TC.labels'):
    file_names, labels, spans = [], [], []
    with open(file, 'r') as label_file:
        for line in label_file:
            split_line = line.split('\t')
            file_names.append(split_line[0])
            labels.append(task2_labels[split_line[1]])
            spans.append((int(split_line[2]), int(split_line[3])))

    return file_names, labels, spans


def label_data(corpus, corpus_labels, tokenizer='punct'):
    samples, labels, spans, file_names = [], [], [], []
    tknz = tree_bank_tokenizer() if tokenizer == 'treebank' else WordPunctTokenizer()
    print('Preprocessing')
    regex = get_regex()
    for article_name in corpus:
        article_propaganda_spans = iter([])
        if corpus_labels:
            article_propaganda_spans = iter(sorted(corpus_labels[article_name])) if article_name in corpus_labels else iter([])
        propaganda_span = next(article_propaganda_spans, [])
        split_article = corpus[article_name].split('\n')
        index_offset = 0
        first_in_span = True
        for line in split_article:
            if not line:
                index_offset += 1
                continue
            line = re.sub(regex, lambda sent:  '~' * len(sent.group()), line)
            line = sub_unicode_chars(line)
            tokenized_line = tknz.tokenize(line)
            span_tokenized_line = tknz.span_tokenize(line)
            line_spans, line_labels = [], []
            for _ in tokenized_line:
                span_wo_offset = next(span_tokenized_line)
                current_span = (span_wo_offset[0] + index_offset, span_wo_offset[1] + index_offset)
                if propaganda_span and current_span[0] >= propaganda_span[1]:
                    propaganda_span = next(article_propaganda_spans, [])
                    first_in_span = True
                if propaganda_span and propaganda_span[0] <= current_span[0] < propaganda_span[1]:
                    line_labels.append(1 if first_in_span else 2)
                    first_in_span = False
                else:
                    line_labels.append(0)
                line_spans.append(current_span)
            assert len(tokenized_line) == len(line_labels) == len(
                line_spans), 'Number of tokens is not equal to the number of spans or labels'
            labels.append(line_labels)
            spans.append(line_spans)
            samples.append(tokenized_line)
            file_names.append(article_name)
            index_offset += len(line) + 1
    print('Preprocessing done')
    return samples, labels, spans, file_names


def tree_bank_tokenizer():
    tokenizer = TreebankWordTokenizer()
    tokenizer.PUNCTUATION.append((re.compile(r'[/\-]'), r' \g<0> '))
    tokenizer.PUNCTUATION.append((re.compile(r'\.\.'), r' .. '))
    tokenizer.PUNCTUATION.append((re.compile(r'[\.,\+]'), r' \g<0> '))
    tokenizer.STARTING_QUOTES.append(
        (re.compile(r"(')(?![sS]\s|[mM]\s|[dD]\s|ll\s|LL\s|re\s|RE\s|ve\s|VE\s|t\s|T\s|\s)"), r" \1 "))
    return tokenizer


def load_task2(articles_path, labels_path, tokenizer='punct'):
    file_names, labels, spans = get_class_labels(labels_path)
    corpus = load_data(articles_path)
    tknz = WordPunctTokenizer()
    samples = []
    for span, file_name in zip(spans, file_names):
        article = corpus[file_name]
        tokenized_span = tknz.tokenize(article[span[0]:span[1]])
        samples.append(tokenized_span)
    return samples, labels, spans, file_names

def load_task2_bert(articles_path, labels_path, tokenizer='punct'):
    file_names, labels, spans = get_class_labels(labels_path)
    corpus = load_data(articles_path)
    samples = []
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for span, file_name in zip(spans, file_names):
        article = corpus[file_name]
        tokenized_sent = bert_tokenizer.tokenize(article[span[0]:span[1]])
        samples.append(tokenized_sent)
    return samples, labels, spans, file_names


def remove_tokens(samples, labels, spans, file_names):
    regex = '~{3,}'
    for sen_toks, sen_labs, sen_spans in zip(samples, labels, spans):
        idx = []
        for i in range(len(sen_toks)):
            if re.search(regex, sen_toks[i]):
                idx.append(i)
        for i in reversed(idx):
            del sen_toks[i]
            del sen_labs[i]
            del sen_spans[i]
    idx = []
    for i in range(len(samples)):
        if not samples[i]:
            idx.append(i)

    for i in reversed(idx):
        del samples[i]
        del labels[i]
        del spans[i]
        del file_names[i]
    return samples, labels, spans, file_names


def load_task1(articles_path, labels_path, tokenizer):
    corpus_labels = get_span_labels(labels_path) if labels_path else None
    corpus = load_data(articles_path)
    samples, labels, spans, file_names = label_data(corpus, corpus_labels, tokenizer)
    labels = list(map(lambda sen: [0 if x == 0 else 1 for x in sen], labels))
    samples, labels, spans, file_names = remove_tokens(samples, labels, spans, file_names)
    return samples, labels, spans, file_names


