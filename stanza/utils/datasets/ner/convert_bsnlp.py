import argparse
import glob
import os
import random
import re

import stanza

AVAILABLE_LANGUAGES = ("bg", "cs", "pl", "ru")

def normalize_entity(text, entity, raw):
    entity = entity.strip()
    # sanity check that the token is in the original text
    if text.find(entity) >= 0:
        return entity

    # some entities have quotes, but the quotes are different from those in the data file
    # for example:
    #   'Съвет "Общи въпроси"'
    #   /home/john/extern_data/ner/bsnlp2019/training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1002.txt
    if sum(1 for x in entity if x == '"') == 2:
        quote_entity = entity.replace('"', '„', 1).replace('"', '“')
        if text.find(quote_entity) >= 0:
            print("Warning: searching for '%s' instead of '%s' in %s" % (quote_entity, entity, raw))
            return quote_entity

    if sum(1 for x in entity if x == '"') == 1:
        quote_entity = entity.replace('"', '„', 1)
        if text.find(quote_entity) >= 0:
            print("Warning: searching for '%s' instead of '%s' in %s" % (quote_entity, entity, raw))
            return quote_entity

    substitution_pairs = {
        # this exact error happens in training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1065.txt
        'Харолд Уилсон': 'Харолд Уилсън',
        # this exact error also happens in training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1065.txt
        'Манчестърски университет': 'Манчестърския университет',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1096.txt
        'Обединеното кралство в променящата се Европа': 'Обединеното кралство в променяща се Европа',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1096.txt
        'Иранската ядрена програма': 'иранската ядрена програма',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1160.txt
        'Представителството на ЕК в българия': 'Представителството на ЕК в България',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1168.txt
        'лейбъристите': 'Лейбъристите',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1175.txt
        'The Daily Express': 'Daily Express',
        # TODO: there are a lot more after this, but at this point I
        # gave up and just emailed the organizers
    }

    if entity in substitution_pairs and text.find(substitution_pairs[entity]) >= 0:
        fixed_entity = substitution_pairs[entity]
        print("Warning: searching for '%s' instead of '%s' in %s" % (fixed_entity, entity, raw))
        return fixed_entity

    # oops, can't find it anywhere
    # want to raise ValueError but there are just too many in the train set for BG
    print("ERROR: Could not find '%s' in %s" % (entity, raw))


def get_sentences(pipeline, annotated, raw):
    annotated_sentences = []
    with open(raw) as fin:
        lines = fin.readlines()
    if len(lines) < 5:
        raise ValueError("Unexpected format in %s" % raw)
    text = "\n".join(lines[4:])

    entities = {}
    with open(annotated) as fin:
        # first line
        header = fin.readline().strip()
        if len(header.split("\t")) > 1:
            raise ValueError("Unexpected missing header line in %s" % annotated)
        for line in fin:
            pieces = line.strip().split("\t")
            if len(pieces) < 3 or len(pieces) > 4:
                raise ValueError("Unexpected annotation format in %s" % annotated)

            entity = normalize_entity(text, pieces[0], raw)
            if not entity:
                continue
            if entity in entities:
                if entities[entity] != pieces[2]:
                    # would like to make this an error, but it actually happens and it's not clear how to fix
                    # annotated/nord_stream/bg/nord_stream_bg.txt_file_119.out
                    print("Warning: found multiple definitions for %s in %s" % (pieces[0], annotated))
                    entities[entity] = pieces[2]
            else:
                entities[entity] = pieces[2]

    tokenized = pipeline(text)
    regex = "(" + "|".join(re.escape(x) for x in entities.keys()) + ")"
    regex = re.compile(regex)

    bad_sentences = set()

    for match in regex.finditer(text):
        start_char, end_char = match.span()
        # this is inefficient, but for something only run once, it shouldn't matter
        start_token = None
        end_token = None
        for token in tokenized.iter_tokens():
            # for example, bsnlp2019/raw/nord_stream/bg/nord_stream_bg.txt_file_100.txt
            # the Solitaire gets stuck with the close quote
            if token.start_char == start_char:
                start_token = token
            if token.end_char == end_char:
                end_token = token
                break
        if start_token is None and end_token is None:
            for token in tokenized.iter_tokens():
                # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1161.txt
                # there are / between words, and the tokenizer does not handle that
                if token.start_char <= start_char and token.end_char >= start_char:
                    start_token = token
                if token.start_char <= end_char and token.end_char >= end_char:
                    end_token = token
                    break
            if start_token is None and end_token is None:
                raise RuntimeError("Match %s did not align with any tokens in %s" % (match.group(0), raw))
            else:
                if start_token:
                    bad_sentences.add(start_token.sent.id)
                else:
                    bad_sentences.add(end_token.sent.id)
                print("Warning: match %s matched in the middle of a token in %s" % (match.group(0), raw))
                continue
        if start_token is None:
            bad_sentences.add(end_token.sent.id)
            print("Warning: match %s started matching in the middle of a token in %s" % (match.group(0), raw))
            continue
        if end_token is None:
            bad_sentences.add(start_token.sent.id)
            print("Warning: match %s ended matching in the middle of a token in %s" % (match.group(0), raw))
            continue
        if not start_token.sent is end_token.sent:
            bad_sentences.add(start_token.sent.id)
            bad_sentences.add(end_token.sent.id)
            print("Warning: match %s spanned sentences %d and %d in document %s" % (match.group(0), start_token.sent.id, end_token.sent.id, raw))
            continue
        # ids start at 1, not 0, so we have to subtract 1
        # then the end token is included, so we add back the 1
        # TODO: verify that this is correct if the language has MWE - cs, pl, for example
        tokens = start_token.sent.tokens[start_token.id[0]-1:end_token.id[0]]
        match_text = match.group(0)
        if match_text not in entities:
            raise RuntimeError("Matched %s, which is not in the entities from %s" % (match_text, annotated))
        ner_tag = entities[match_text]
        tokens[0].ner = "B-" + ner_tag
        for token in tokens[1:]:
            token.ner = "I-" + ner_tag

    for sentence in tokenized.sentences:
        if not sentence.id in bad_sentences:
            annotated_sentences.append(sentence)

    return annotated_sentences

def write_sentences(output_filename, annotated_sentences):
    print("Writing %d sentences to %s" % (len(annotated_sentences), output_filename))
    with open(output_filename, "w") as fout:
        for sentence in annotated_sentences:
            for token in sentence.tokens:
                ner_tag = token.ner
                if not ner_tag:
                    ner_tag = "O"
                fout.write("%s\t%s\n" % (token.text, ner_tag))
            fout.write("\n")


def convert_bsnlp(language, base_input_path, output_filename, split_filename=None):
    if language not in AVAILABLE_LANGUAGES:
        raise ValueError("The current BSNLP datasets only include the following languages: %s" % ",".join(AVAILABLE_LANGUAGES))
    pipeline = stanza.Pipeline(language, processors="tokenize")
    random.seed(1234)

    annotated_path = os.path.join(base_input_path, "annotated", "*", language, "*")
    annotated_files = sorted(glob.glob(annotated_path))
    raw_path = os.path.join(base_input_path, "raw", "*", language, "*")
    raw_files = sorted(glob.glob(raw_path))

    if len(annotated_files) == 0 and len(raw_files) == 0:
        print("Could not find files in %s" % annotated_path)
        annotated_path = os.path.join(base_input_path, "annotated", language, "*")
        print("Trying %s instead" % annotated_path)
        annotated_files = sorted(glob.glob(annotated_path))
        raw_path = os.path.join(base_input_path, "raw", language, "*")
        raw_files = sorted(glob.glob(raw_path))

    if len(annotated_files) != len(raw_files):
        raise ValueError("Unexpected differences in the file lists between %s and %s" % (annotated_files, raw_files))

    for i, j in zip(annotated_files, raw_files):
        if os.path.split(i)[1][:-4] != os.path.split(j)[1][:-4]:
            raise ValueError("Unexpected differences in the file lists: found %s instead of %s" % (i, j))

    annotated_sentences = []
    if split_filename:
        split_sentences = []
    for annotated, raw in zip(annotated_files, raw_files):
        new_sentences = get_sentences(pipeline, annotated, raw)
        if not split_filename or random.random() < 0.85:
            annotated_sentences.extend(new_sentences)
        else:
            split_sentences.extend(new_sentences)

    write_sentences(output_filename, annotated_sentences)
    if split_filename:
        write_sentences(split_filename, split_sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default="bg", help="Language to process")
    parser.add_argument('--input_path', type=str, default="/home/john/extern_data/ner/bsnlp2019", help="Where to find the files")
    parser.add_argument('--output_path', type=str, default="/home/john/stanza/data/ner/bg_bsnlp.test.csv", help="Where to output the results")
    parser.add_argument('--dev_path', type=str, default=None, help="A secondary output path - 15% of the data will go here")
    args = parser.parse_args()

    convert_bsnlp(args.language, args.input_path, args.output_path, args.dev_path)
