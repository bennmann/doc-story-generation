from story_generation.common.summarizer.models.gpt3_summarizer import GPT3Summarizer
from story_generation.common.summarizer.models.opt_summarizer import OPTSummarizer
from story_generation.common.summarizer.models.rwkv_summarizer import RWKVSummarizer

SUMMARIZER_CHOICES=['gpt3_summarizer', 'opt_summarizer', 'rwkv_summarizer']

def add_summarizer_args(parser):
    parser.add_argument('--summarizer', type=str, default='rwkv_summarizer', choices=SUMMARIZER_CHOICES, help='model architecture')
    parser.add_argument('--summarizer-save-dir', type=str, default=None, help='directory to save summarizer')
    parser.add_argument('--summarizer-load-dir', type=str, default=None, help='directory to load summarizer')
    parser.add_argument('--summarizer-temperature', type=float, default=0.8, help='temperature for summarizer')
    parser.add_argument('--opt-summarizer-temperature', type=float, default=0.8, help='temperature for OPT summarizer during main story generation')
    parser.add_argument('--summarizer-top-p', type=float, default=1.0, help='top p for summarizer')
    parser.add_argument('--summarizer-top-k', type=float, default=100, help='top k for summarizer if supported')
    parser.add_argument('--summarizer-prompt-penalty', type=float, default=1, help='OPT control penalty for prompt tokens for summarizer, excluding stopwords/punc/names')
    parser.add_argument('--summarizer-frequency-penalty', type=float, default=1, help='frequency penalty for summarizer')
    parser.add_argument('--summarizer-presence-penalty', type=float, default=0, help='presence penalty for summarizer')
    parser.add_argument('--summarizer-frequency-penalty-decay', type=float, default=0.98, help='frequency penalty decay for OPT summarizer')
    parser.add_argument('--max-tokens', type=int, default=64, help='max length for generation, not including prompt')
    parser.add_argument('--gpt3-model', type=str, default='text-davinci-002', help='gpt3 model or finetuned ckpt for GPT3Summarizer')
    parser.add_argument('--max-context-length', type=int, default=1024, help='max length for context to facilitate toy version')
    parser.add_argument('--alpa-url', type=str, default=None, help='url for alpa API')
    parser.add_argument('--alpa-port', type=str, default=None, help='port for alpa API, if alpa-url is a filename to read server location from. convenient for slurm')
    parser.add_argument('--alpa-key', type=str, default='', help='key for alpa API, if using the public API')
    parser.add_argument('--rwkv_model', type=str, default='rwkv_summarizer', help='On Premise/local model architecture based on RNN language model (search online for RWKV)'
    return parser

def load_summarizer(args):
    if args.summarizer == 'gpt3_summarizer':
        summarizer = GPT3Summarizer(args)
    elif args.summarizer == 'opt_summarizer':
        summarizer = OPTSummarizer(args)
    elif args.summarizer == 'rwkv_summarizer':
        summarizer = RWKVSummarizer(args)
    else:
        raise NotImplementedError
    return summarizer
