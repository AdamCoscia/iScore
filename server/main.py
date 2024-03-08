import os
import string
from pathlib import Path

punctuation = set(string.punctuation)
os.environ["TRANSFORMERS_CACHE"] = "./transformers/"


#
# SymSpell packages
#
import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


#
# NLTK packages
#
import nltk

file_dir_path = os.path.dirname(os.path.realpath(__file__))
nltk_data_dir_path = os.path.join(file_dir_path, "nltk")

Path(nltk_data_dir_path).mkdir(parents=True, exist_ok=True)

nltk.data.path.append(nltk_data_dir_path)
nltk.download("punkt", download_dir=nltk_data_dir_path)
nltk.download("stopwords", download_dir=nltk_data_dir_path)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir_path)
nltk.download("wordnet", download_dir=nltk_data_dir_path)
nltk.download("omw-1.4", download_dir=nltk_data_dir_path)

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


#
# PyTorch packages
#
import torch

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"


#
# Transformers packages
#
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

tok_mod = {
    "slc": {
        "tokenizer": AutoTokenizer.from_pretrained("tiedaar/summary-longformer-content"),
        "model_default": AutoModelForSequenceClassification.from_pretrained(
            "tiedaar/summary-longformer-content", num_labels=1
        ),
    },
    "lcg": {
        "tokenizer": AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
        "model_default": AutoModelForSequenceClassification.from_pretrained(
            "tiedaar/longformer-content-global", num_labels=1
        ),
    },
    "slw": {
        "tokenizer": AutoTokenizer.from_pretrained("tiedaar/summary-longformer-wording"),
        "model_default": AutoModelForSequenceClassification.from_pretrained(
            "tiedaar/summary-longformer-wording", num_labels=1
        ),
    },
    "lwg": {
        "tokenizer": AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
        "model_default": AutoModelForSequenceClassification.from_pretrained(
            "tiedaar/longformer-wording-global", num_labels=1
        ),
    },
    # "keyphrases": {
    #     "tokenizer": AutoTokenizer.from_pretrained("bloomberg/KeyBART"),
    #     "model_default": AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART"),
    # },
}

tok_mod["slc"]["model_default"].eval()
tok_mod["lcg"]["model_default"].eval()
tok_mod["slw"]["model_default"].eval()
tok_mod["lwg"]["model_default"].eval()
# tok_mod["keyphrases"]["model_default"].eval()

#
# Data processing packges
#
import numpy as np
import pandas as pd

options = {}


def pos_to_wordnet_pos(penntag, returnNone=False):
    """Mapping from POS tag word wordnet pos tag.

    See: <https://stackoverflow.com/a/63667489>
    """
    morphy_tag = {"NN": wn.NOUN, "JJ": wn.ADJ, "VB": wn.VERB, "RB": wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ""


def get_synonyms(word, pos):
    """Gets word synonyms for part of speech.

    See: <https://stackoverflow.com/a/63667489>
    """
    for synset in wn.synsets(word, pos=pos_to_wordnet_pos(pos)):
        for lemma in synset.lemmas():
            yield lemma.name()


def reshape_longformer_local_attentions(model_output):
    """Reshape Longformer local attention scores for chunking in response to client.

    See: <https://huggingface.co/docs/transformers/v4.29.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput>

    How to index between `a` and `a_list`:
    ```
    a.shape = [layers, heads, seq_len, x + attention_window + 1]
    i = (layer * a.shape[0] * a.shape[2]) + (head * a.shape[2]) + token
    a_list[i] ==  a[layer, head, token, :]

    e.g.,
    layers = 12, heads = 12, seq_len = 712, x = 1, attention_window = 512
    layer = 2, head = 1, token_from = 1
    i = (2 * 12 * 712) + (1 * 712) + 1 = 17801
    a_list[17801] ==  a[2, 1, 1, :]
    len(a_list[17801]) == a.shape[3])
    ```
    """
    attentions = model_output["attentions"]
    a = torch.stack(attentions)[:, 0, :, :, :].detach().cpu().numpy()
    a_shape = a.shape  # tuple
    a_flat = a.ravel()
    a_list = np.split(a_flat, a_flat.shape[0] / a_shape[3])  # python list of numpy arrays
    model_output["local_attentions_shape"] = a_shape
    model_output["local_attentions_list"] = a_list


def reshape_longformer_global_attentions(model_output):
    """Reshape Longformer global attention scores for chunking in response to client.

    See: <https://huggingface.co/docs/transformers/v4.29.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput>

    How to index between `b` and `b_list`:
    ```
    b.shape = [layers, heads, seq_len, x]
    i = (layer * b.shape[0] * b.shape[2]) + (head * b.shape[2]) + token
    b_list[i] ==  a[layer, head, token, :]

    e.g.,
    layers = 12, heads = 12, seq_len = 1024, x = 1
    layer = 2, head = 1, token_to = 1
    i = (2 * 12 * 1024) + (1 * 1024) + 1 = 25601
    b_list[25601] == b[2, 1, 1, :]
    len(b_list[25601]) == b.shape[3])
    ```
    """
    global_attentions = model_output["global_attentions"]
    b = torch.stack(global_attentions)[:, 0, :, :, :].detach().cpu().numpy()
    b_shape = b.shape  # tuple
    b_flat = b.ravel()
    b_list = np.split(b_flat, b_flat.shape[0] / b_shape[3])  # python list of numpy arrays
    model_output["global_attentions_shape"] = b_shape
    model_output["global_attentions_list"] = b_list


def compute_perturbation_scores(inputs, model_output):
    """Compute new scores by perturbing the input and diff them with the true score.

    At sentence-level, remove each sentence from entire input and recompute score.

    At word-level, replace nouns and verbs in summary with synonyms and recompute score.

    At token-level, mask attention of each token in the summary and recompute score.
    """

    def infer_longformer(text):
        """Run inference on Longformer models."""
        model = model_output["model"]
        tokenizer = model_output["tokenizer"]
        encoding = tokenizer(text, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        global_attention_mask = torch.zeros_like(input_ids)
        if model_output["model_checkpoint"] in ["lcg", "lwg"]:
            # set all tokens in summary plus </s> token to have global attention
            sep_index = [tokenizer.decode(input_ids[0][i]) for i in range(len(input_ids[0]))].index("</s>")
            global_attention_mask[:, 0 : (sep_index + 1)] = 1
        else:
            # set only first token (<s>) to have global attention
            global_attention_mask[:, 0] = 1
        with torch.no_grad():
            output = model(
                input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask
            )
        score = output["logits"].detach().cpu().numpy()[0][0]
        return score

    def contains_punctuation(text):
        """Returns true if text contains no punctuation."""
        for punct in punctuation:
            if punct in text:
                return True
        return False

    grammar_out = []
    sentences_out = []
    words_out = []
    words_nested = []
    tokens_out = []
    input_text = inputs["summary"] + "</s>" + inputs["source"]

    # Compute score with grammar fixed
    if options[inputs["id"]]["get_gram_pert"]:
        print("      fixing grammar...")
        grammars = []
        types = []
        scores = []

        # add original input and score for client to render
        grammars.append(inputs["summary"])
        types.append("grammar")
        scores.append(model_output["true_score"])

        # add line break for client to render
        grammars.append("<br />")
        types.append("break")
        scores.append(np.nan)

        # use SymSpell to correct spelling of summary (keep punctuation and casing)
        summary = inputs["summary"]
        summary_corrected = ""
        summary_token_spans = list(nltk.tokenize.TreebankWordTokenizer().span_tokenize(summary))
        prev_end = 0
        for start, end in summary_token_spans:
            if prev_end != start:
                summary_corrected += summary[prev_end:start]
            summary_word = summary[start:end]
            if summary_word not in stop_words and not contains_punctuation(summary_word):
                suggestions = sym_spell.lookup(
                    summary_word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, transfer_casing=True
                )
                corrected_word = suggestions[0].term
                summary_corrected += corrected_word
            else:
                summary_corrected += summary_word
            prev_end = end
        g = summary_corrected + "</s>" + inputs["source"]
        score = infer_longformer(g)
        grammars.append(summary_corrected)
        types.append("grammar")
        scores.append(score)

        # add line break for client to render
        grammars.append("<br />")
        types.append("break")
        scores.append(np.nan)

        # use SymSpell to correct spelling of summary (remove punctuation and casing)
        suggestions = sym_spell.lookup_compound(inputs["summary"], max_edit_distance=2)
        summary_corrected = suggestions[0].term
        g = summary_corrected + "</s>" + inputs["source"]
        score = infer_longformer(g)
        grammars.append(summary_corrected)
        types.append("grammar")
        scores.append(score)

        # add line break for client to render
        grammars.append("<br />")
        types.append("break")
        scores.append(np.nan)

        # use SymSpell to correct spelling of summary (segment words)
        result = sym_spell.word_segmentation(inputs["summary"])
        summary_corrected = result.corrected_string
        g = summary_corrected + "</s>" + inputs["source"]
        score = infer_longformer(g)
        grammars.append(summary_corrected)
        types.append("grammar")
        scores.append(score)

        # compute differences between score and true score
        df_grammar = pd.DataFrame(list(zip(grammars, types, scores)), columns=["grammar", "type", "score"])
        df_grammar["diff_true"] = df_grammar["score"] - model_output["true_score"]
        df_grammar["diff_abs"] = df_grammar["score"].apply(lambda x: abs(x - model_output["true_score"]))
        df_grammar["diff_sum_norm"] = df_grammar["diff_abs"] / df_grammar["diff_abs"].sum()
        df_grammar["diff_max_norm"] = df_grammar["diff_abs"] / df_grammar["diff_abs"].max()
        grammar_out = df_grammar.to_dict(orient="records")

    # Compute scores with each sentence removed from the entire input
    if options[inputs["id"]]["get_sent_pert"]:
        print("      removing sentences...")
        sentences = []
        types = []
        scores = []

        # replace summary sentences
        summary_sentences = nltk.tokenize.sent_tokenize(inputs["summary"])
        n_summary_sentences = len(summary_sentences)
        for i in range(n_summary_sentences):
            print(f"      {i + 1}/{n_summary_sentences - 1}", end="\r")
            s = summary_sentences[i]
            x = input_text.replace(s, "")  # remove sentence from input
            score = infer_longformer(x)
            sentences.append(s)
            types.append("summary")
            scores.append(score)
        print(" " * 30, end="\r")

        #  add line break for the client to render
        sentences.append("<br />")
        types.append("break")
        scores.append(np.nan)

        # replace source sentences
        source_sentences = nltk.tokenize.sent_tokenize(inputs["source"])
        n_source_sentences = len(source_sentences)
        for i in range(n_source_sentences):
            print(f"      {i + 1}/{n_source_sentences - 1}", end="\r")
            s = source_sentences[i]
            x = input_text.replace(s, "")  # remove sentence from input
            score = infer_longformer(x)
            sentences.append(s)
            types.append("source")
            scores.append(score)
        print(" " * 30, end="\r")

        # compute differences between score and true score
        df_sentences = pd.DataFrame(list(zip(sentences, types, scores)), columns=["sentence", "type", "score"])
        df_sentences["diff_true"] = df_sentences["score"] - model_output["true_score"]
        df_sentences["diff_abs"] = df_sentences["score"].apply(lambda x: abs(x - model_output["true_score"]))
        df_sentences["diff_sum_norm"] = df_sentences["diff_abs"] / df_sentences["diff_abs"].sum()
        df_sentences["diff_max_norm"] = df_sentences["diff_abs"] / df_sentences["diff_abs"].max()
        sentences_out = df_sentences.to_dict(orient="records")

    # Compute scores with each noun, verb in summary replaced with synonym
    if options[inputs["id"]]["get_word_pert"]:
        print("      replacing words...")
        summary_words = nltk.word_tokenize(inputs["summary"])
        summary_word_synonyms = []
        for word, tag in nltk.pos_tag(summary_words):
            word_synonyms = []
            if word not in stop_words and not contains_punctuation(word):
                word_synonyms = sorted(set(synonym for synonym in get_synonyms(word, tag) if synonym != word))
            summary_word_synonyms.append((word, word_synonyms))
        n_summary_words = len(summary_word_synonyms)

        indices = []
        words = []
        synonyms = []
        scores = []
        for i in range(n_summary_words):
            print(f"      {i + 1}/{n_summary_words - 1}", end="\r")
            word = summary_word_synonyms[i][0]
            word_synonyms = summary_word_synonyms[i][1]
            n_word_synonyms = len(word_synonyms)
            if n_word_synonyms < 1:
                indices.append(i)
                words.append(word)
                synonyms.append(None)
                scores.append(np.nan)
            else:
                for j in range(n_word_synonyms):
                    print(f"        {j + 1}/{n_word_synonyms - 1}", end="\r")
                    synonym = word_synonyms[j].replace("_", " ")  # replace underscores with whitespaces
                    x = input_text.replace(word, synonym)  # replace word with synonym in input
                    score = infer_longformer(x)
                    indices.append(i)
                    words.append(word)
                    synonyms.append(synonym)
                    scores.append(score)
                print(" " * 30, end="\r")
            print(" " * 30, end="\r")
        print(" " * 30, end="\r")

        df_words = pd.DataFrame(
            list(zip(indices, words, synonyms, scores)), columns=["idx", "word", "synonym", "score"]
        )
        df_words["diff_true"] = df_words["score"] - model_output["true_score"]
        df_words["diff_abs"] = df_words["score"].apply(lambda x: abs(x - model_output["true_score"]))
        df_words["diff_sum_norm"] = df_words["diff_abs"] / df_words["diff_abs"].sum()
        df_words["diff_max_norm"] = df_words["diff_abs"] / df_words["diff_abs"].max()
        words_out = df_words.to_dict(orient="records")

        # nest synonyms by word, where empty list == no synonyms
        helper = {}
        for r in words_out:
            i = r["idx"]
            w = r["word"]
            w_has_syn = not r["synonym"] is None
            if (i, w) in helper:
                if w_has_syn:
                    helper[(i, w)].append(r)
            else:
                if w_has_syn:
                    helper[(i, w)] = [r]
                else:
                    helper[(i, w)] = []
        words_nested = [{"word": k[1], "_children": v} for k, v in helper.items()]

    # Compute scores with each token masked from the summary
    if options[inputs["id"]]["get_token_pert"]:
        print("      masking tokens...")
        model = model_output["model"]
        tokenizer = model_output["tokenizer"]

        encoding = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        global_attention_mask = torch.zeros_like(input_ids)
        if model_output["model_checkpoint"] in ["lcg", "lwg"]:
            # set all tokens in summary plus </s> token to have global attention
            sep_index = [tokenizer.decode(input_ids[0][i]) for i in range(len(input_ids[0]))].index("</s>")
            global_attention_mask[:, 0 : (sep_index + 1)] = 1
        else:
            # set only first token (<s>) to have global attention
            global_attention_mask[:, 0] = 1

        n_summary_tokens = tokenizer(inputs["summary"], return_tensors="pt")["input_ids"].detach().numpy().shape[1]
        tokens = []
        scores = []
        for i in range(1, n_summary_tokens - 1):
            print(f"      {i + 1}/{n_summary_tokens - 1}", end="\r")
            token_id = input_ids[:, i].cpu().detach().numpy()[0]  # get token id
            token = tokenizer.decode(token_id)  # get token as word
            attention_mask_set = torch.clone(attention_mask.detach())  # create copy of attention mask
            attention_mask_set[:, i] = 0  # flip single mask bit to 0 at index
            with torch.no_grad():
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask_set, global_attention_mask=global_attention_mask
                )
            score = output["logits"].detach().cpu().numpy()[0][0]
            tokens.append(token)
            scores.append(score)
        print(" " * 30, end="\r")

        df_tokens = pd.DataFrame(list(zip(tokens, scores)), columns=["token", "score"])
        df_tokens["diff_true"] = df_tokens["score"] - model_output["true_score"]
        df_tokens["diff_abs"] = df_tokens["score"].apply(lambda x: abs(x - model_output["true_score"]))
        df_tokens["diff_sum_norm"] = df_tokens["diff_abs"] / df_tokens["diff_abs"].sum()
        df_tokens["diff_max_norm"] = df_tokens["diff_abs"] / df_tokens["diff_abs"].max()
        tokens_out = df_tokens.to_dict(orient="records")

    model_output["perturbation_scores"] = {
        "grammar": grammar_out,
        "sentence": sentences_out,
        "word": {"out": words_out, "nested": words_nested},
        "token": tokens_out,
    }


def compute_analytic_scores(inputs):
    """Return summary evlauation scores based on summary and source text."""

    def infer_longformer(model_checkpoint, text):
        """Run inference on Longformer models."""
        model_default = tok_mod[model_checkpoint]["model_default"]
        model = model_default.to(device)
        tokenizer = tok_mod[model_checkpoint]["tokenizer"]
        encoding = tokenizer(text, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        global_attention_mask = torch.zeros_like(input_ids)
        token_ids = input_ids.detach().cpu().numpy()[0]
        tokens = [(i, token_ids[i], tokenizer.decode(token_ids[i])) for i in range(len(token_ids))]
        global_tokens = []
        if model_checkpoint in ["lcg", "lwg"]:
            # set all tokens in summary plus </s> token to have global attention
            sep_index = [tokenizer.decode(input_ids[0][i]) for i in range(len(input_ids[0]))].index("</s>")
            global_attention_mask[:, 0 : (sep_index + 1)] = 1
            global_tokens = list(range(0, sep_index + 1))
        else:
            # set only first token (<s>) to have global attention
            global_attention_mask[:, 0] = 1
            global_tokens = [0]
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
        true_score = output["logits"].detach().cpu().numpy()[0][0]
        attentions = output["attentions"]
        global_attentions = output["global_attentions"]
        hidden_states = output["hidden_states"]
        return {
            "model_checkpoint": model_checkpoint,
            "model": model,
            "tokenizer": tokenizer,
            "tokens": tokens,
            "global_tokens": global_tokens,
            "true_score": true_score,
            "attentions": attentions,
            "global_attentions": global_attentions,
            "hidden_states": hidden_states,
        }

    input_text = inputs["summary"] + "</s>" + inputs["source"]
    model_names = inputs["models"]
    out = {}

    # models:
    # slc == summary-longformer-content
    # slw == summary-longformer-wording
    # lcg == longformer-content-global
    # lwg == longformer-wording-global

    for model_name in model_names:
        print(f"    running '{model_name}' inference...")
        model_out = infer_longformer(model_name, input_text)

        print(f"    reshaping '{model_name}' attentions...")
        reshape_longformer_local_attentions(model_out)
        reshape_longformer_global_attentions(model_out)

        print(f"    computing '{model_name}' perturbation scores...")
        compute_perturbation_scores(inputs, model_out)

        out[f"{model_name}_tokens"] = model_out["tokens"]
        out[f"{model_name}_global_tokens"] = model_out["global_tokens"]
        out[f"{model_name}_score"] = model_out["true_score"]
        out[f"{model_name}_perturbation_scores"] = model_out["perturbation_scores"]
        out[f"{model_name}_local_attentions_shape"] = model_out["local_attentions_shape"]
        out[f"{model_name}_global_attentions_shape"] = model_out["global_attentions_shape"]
        out[f"{model_name}_local_attentions_list"] = model_out["local_attentions_list"]
        out[f"{model_name}_global_attentions_list"] = model_out["global_attentions_list"]

    return out


def compute_keyphrases(inputs):
    """Return keyphrases from summary and source text.

    See: <https://huggingface.co/bloomberg/KeyBART>
    """
    model_checkpoint = "keyphrases"
    model_default = tok_mod[model_checkpoint]["model_default"]
    model = model_default.to(device)
    tokenizer = tok_mod[model_checkpoint]["tokenizer"]

    # Source
    source_encoding = tokenizer(inputs["source"], return_tensors="pt").to(device)
    input_ids = source_encoding["input_ids"][:, :1024]  # take up to first 1024 tokens (max seq len for KeyBART)
    output_ids = model.generate(input_ids, num_beams=2, min_length=0, max_length=20)
    source_keyphrases = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {"source_keyphrases": source_keyphrases}


def summary_score(inputs):
    """Score a source/summary pair using HuggingFace transformers.

    inputs <- { source, summary }
    """
    print("  tokenizing input...")
    summary_sentences = nltk.tokenize.sent_tokenize(inputs["summary"])
    source_sentences = nltk.tokenize.sent_tokenize(inputs["source"])
    summary_words = nltk.word_tokenize(inputs["summary"])
    source_words = nltk.word_tokenize(inputs["source"])

    print("  computing analytic scores...")
    analytic_scores = compute_analytic_scores(inputs)

    # print("  computing keyphrases...")
    # keyphrases = compute_keyphrases(inputs)

    # MUST BE IN THE SAME ORDER THAT THE INTERFACE RECEIVES THEM
    dicts = []
    lists = []

    inputs_out = {
        "summary_input": {
            "id": inputs["id"],
            "text": inputs["summary"],
            "sentences": summary_sentences,
            "words": summary_words,
        },
        "source_input": {
            "text": inputs["source"],
            "sentences": source_sentences,
            "words": source_words,
        },
    }
    # keyphrases_out = {
    #     "source_keyphrases": keyphrases["source_keyphrases"],
    # }
    scores = {}
    tokens = {}
    p_scores = {}
    a_shapes = {}
    model_names = inputs["models"]
    for model_name in model_names:
        tokens[f"{model_name}_tokens"] = analytic_scores[f"{model_name}_tokens"]
        tokens[f"{model_name}_global_tokens"] = analytic_scores[f"{model_name}_global_tokens"]
        scores[f"{model_name}_score"] = analytic_scores[f"{model_name}_score"]
        p_scores[f"{model_name}_perturbation_scores"] = analytic_scores[f"{model_name}_perturbation_scores"]
        a_shapes[f"{model_name}_local_attentions_shape"] = analytic_scores[f"{model_name}_local_attentions_shape"]
        a_shapes[f"{model_name}_global_attentions_shape"] = analytic_scores[f"{model_name}_global_attentions_shape"]
        lists.append(analytic_scores[f"{model_name}_local_attentions_list"])
        lists.append(analytic_scores[f"{model_name}_global_attentions_list"])

    dicts = [
        inputs_out,
        scores,
        # keyphrases_out,
        tokens,
        p_scores,
        a_shapes,
    ]

    return {"dicts": dicts, "lists": lists}


#
# Web app packages
#
import orjson
from flask import Flask, stream_with_context, request, Response
from flask_cors import CORS


# Create Flask app
app = Flask(__name__)
CORS(app)


@app.route("/connect", methods=["GET"])
def connect():
    return "Connected!"


@app.route("/get-data", methods=["POST"])
def get_data():
    """request is sent as JSON, which is converted to a dict. results are
    serialized with custom orjson provider class (above).
    """

    def generate_response(data_out):
        for rid, run_out in data_out.items():
            for d in run_out["dicts"]:
                yield orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"
            if options[rid]["get_attn_scores"]:
                for l in run_out["lists"]:
                    for a in l:
                        yield orjson.dumps(a, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"

    # `data_in` is list of:
    # {
    #   models                   list | model names to run, e.g., "slc"
    #   id                        int | id of summary
    #   source                    str | source text
    #   summary                   str | summary text
    #   getGrammarPerturbation   bool | perturb input by fixing grammar
    #   getSentencePerturbation  bool | perturb input by removing sentences
    #   getWordPerturbation      bool | perturb input by replacing words
    #   getTokenPerturbation     bool | perturb input by masking tokens
    #   getAttentionScores       bool | stream attention back to client
    # }

    data_in = request.json
    data_out = {}

    for run_in in data_in:
        print()
        print(f"summary {run_in['id']} - generating scores...")
        rid = run_in["id"]
        options[rid] = {}
        options[rid]["get_gram_pert"] = run_in["getGrammarPerturbation"] == "true"
        options[rid]["get_sent_pert"] = run_in["getSentencePerturbation"] == "true"
        options[rid]["get_word_pert"] = run_in["getWordPerturbation"] == "true"
        options[rid]["get_token_pert"] = run_in["getTokenPerturbation"] == "true"
        options[rid]["get_attn_scores"] = run_in["getAttentionScores"] == "true"
        run_out = summary_score(run_in)
        data_out[rid] = run_out

    # print("writing data to file...")
    # with open("../interface/assets/data/sampleQuery.js", "w", encoding="utf-8") as f:
    #     f.write("export default ")
    #     f.write(orjson.dumps(data_out, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8", "replace"))

    print("serializing data and streaming...")
    return Response(stream_with_context(generate_response(data_out)))


if __name__ == "__main__":
    app.run(host="localhost", port=int(os.environ.get("PORT", 8001)))
