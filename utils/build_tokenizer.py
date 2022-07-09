from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
language = 'en'

tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
trainer = WordLevelTrainer(min_frequency=10, special_tokens=[SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN])
tokenizer.pre_tokenizer = Whitespace()

raw_dataset = load_dataset("opus_books", "en-fr")
dataset = raw_dataset['train']['translation']


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        tmp = dataset[i:i+batch_size]
        yield [example[language] for example in tmp]


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
tokenizer.post_processor = TemplateProcessing(
    single=f"{SOS_TOKEN} $0 {EOS_TOKEN}",
    pair=f"{SOS_TOKEN} $A {EOS_TOKEN}:0 {SOS_TOKEN}:1 $B:1 {EOS_TOKEN}:1",
    special_tokens=[(SOS_TOKEN, 0), (EOS_TOKEN, 1)]
)
tokenizer.enable_padding(pad_id=3, pad_token=PAD_TOKEN)

pretrained_tokenizer = PreTrainedTokenizerFast(
        bos_token=SOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        tokenizer_object=tokenizer
).save_pretrained(
    save_directory=f'./seq2seq/{language}_tokenizer',
)
