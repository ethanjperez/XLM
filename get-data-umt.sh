# Usage: ./get-data-umt.sh --src mh --tgt sh --reload_codes dumped/xlm_en/codes_en --reload_vocab dumped/xlm_en/vocab_en
# This script will successively:
# 1) download Moses scripts, download and compile fastBPE
# 2) download, extract, tokenize, apply BPE to monolingual and parallel test data
# 3) binarize all datasets

set -e


#
# Data preprocessing configuration
#
N_MONO=18920    # number of monolingual sentences for each language
CODES=30000     # number of BPE codes


#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --reload_codes)
    RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
    RELOAD_VOCAB="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"
DATA_FOLDER=comparison.paired  # TODO: Set via command line

#
# Check parameters
#
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi
if [ "$TGT" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$SRC" != "mh" -a "$SRC" != "sh" ]; then echo "unknown source language"; exit; fi
if [ "$TGT" != "mh" -a "$TGT" != "sh" ]; then echo "unknown target language"; exit; fi
if [ "$SRC" == "$TGT" ]; then echo "source and target cannot be identical"; exit; fi
if [ "$SRC" \> "$TGT" ]; then echo "please ensure SRC < TGT"; exit; fi
if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi


#
# Initialize tools and data paths
#

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
DATA_PATH=$PWD/data/umt/$DATA_FOLDER
PROC_PATH=$DATA_PATH/processed

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $PROC_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# raw, tokenized, and BPE data files
SRC_TRAIN_RAW=$DATA_PATH/train.$SRC
TGT_TRAIN_RAW=$DATA_PATH/train.$TGT
SRC_VALID_RAW=$DATA_PATH/valid.$SRC
TGT_VALID_RAW=$DATA_PATH/valid.$TGT
SRC_TRAIN_TOK=$SRC_TRAIN_RAW.tok
TGT_TRAIN_TOK=$TGT_TRAIN_RAW.tok
SRC_VALID_TOK=$SRC_VALID_RAW.tok
TGT_VALID_TOK=$TGT_VALID_RAW.tok
SRC_TRAIN_BPE=$PROC_PATH/train.$SRC
TGT_TRAIN_BPE=$PROC_PATH/train.$TGT
SRC_VALID_BPE=$PROC_PATH/valid.$SRC
TGT_VALID_BPE=$PROC_PATH/valid.$TGT

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

# install tools
./install-tools.sh

# tokenize data
if ! [[ -f "$SRC_VALID_TOK" ]]; then
  echo "Tokenize $SRC valid monolingual data..."
  eval "cat $SRC_VALID_RAW | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $SRC_VALID_TOK"
fi
if ! [[ -f "$SRC_TRAIN_TOK" ]]; then
  echo "Tokenize $SRC train monolingual data..."
  eval "cat $SRC_TRAIN_RAW | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $SRC_TRAIN_TOK"
fi
if ! [[ -f "$TGT_VALID_TOK" ]]; then
  echo "Tokenize $TGT valid monolingual data..."
  eval "cat $TGT_VALID_RAW | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $TGT_VALID_TOK"
fi
if ! [[ -f "$TGT_TRAIN_TOK" ]]; then
  echo "Tokenize $TGT train monolingual data..."
  eval "cat $TGT_TRAIN_RAW | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $TGT_TRAIN_TOK"
fi
echo "$SRC monolingual data tokenized in: $SRC_TRAIN_TOK and $SRC_VALID_TOK"
echo "$TGT monolingual data tokenized in: $TGT_TRAIN_TOK and $TGT_VALID_TOK"

# reload BPE codes
cd $MAIN_PATH
if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

# reload full vocabulary
cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

# apply BPE codes
if ! [[ -f "$SRC_VALID_BPE" ]]; then
  echo "Applying $SRC BPE codes to valid..."
  $FASTBPE applybpe $SRC_VALID_BPE $SRC_VALID_TOK $BPE_CODES $FULL_VOCAB
fi
if ! [[ -f "$SRC_TRAIN_BPE" ]]; then
  echo "Applying $SRC BPE codes to train..."
  $FASTBPE applybpe $SRC_TRAIN_BPE $SRC_TRAIN_TOK $BPE_CODES $FULL_VOCAB
fi
if ! [[ -f "$TGT_VALID_BPE" ]]; then
  echo "Applying $TGT BPE codes to valid..."
  $FASTBPE applybpe $TGT_VALID_BPE $TGT_VALID_TOK $BPE_CODES $FULL_VOCAB
fi
if ! [[ -f "$TGT_TRAIN_BPE" ]]; then
  echo "Applying $TGT BPE codes to train..."
  $FASTBPE applybpe $TGT_TRAIN_BPE $TGT_TRAIN_TOK $BPE_CODES $FULL_VOCAB
fi
echo "BPE codes applied to $SRC in: $SRC_TRAIN_BPE and $SRC_VALID_BPE"
echo "BPE codes applied to $TGT in: $TGT_TRAIN_BPE and $TGT_VALID_BPE"

# binarize data
if ! [[ -f "$SRC_VALID_BPE.pth" ]]; then
  echo "Binarizing $SRC valid data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_VALID_BPE
fi
if ! [[ -f "$SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $SRC train data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_TRAIN_BPE
fi
if ! [[ -f "$TGT_VALID_BPE.pth" ]]; then
  echo "Binarizing $TGT valid data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TGT_VALID_BPE
fi
if ! [[ -f "$TGT_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $TGT train data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TGT_TRAIN_BPE
fi
echo "$SRC binarized data in: $SRC_TRAIN_BPE.pth and $SRC_VALID_BPE.pth"
echo "$TGT binarized data in: $TGT_TRAIN_BPE.pth and $TGT_VALID_BPE.pth"


#
# Link parallel validation and test data to monolingual data
#
ln -sf $SRC_VALID_BPE.pth $PROC_PATH/valid.$SRC-$TGT.$SRC
ln -sf $TGT_VALID_BPE.pth $PROC_PATH/valid.$SRC-$TGT.$TGT


#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    $SRC: $SRC_TRAIN_BPE.pth"
echo "    $TGT: $TGT_TRAIN_BPE.pth"
echo "Parallel validation data:"
echo "    $SRC: $SRC_VALID_BPE.pth"
echo "    $TGT: $TGT_VALID_BPE.pth"
echo ""
