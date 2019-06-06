# -*- coding: utf-8 -*-
from functools import partial

import six
import itertools
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase
import ast
import pytorch_pretrained_bert as bert
from pytorch_pretrained_bert import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from itertools import repeat 
import torch


embeddingsModel = "bert-base-uncased"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Make a tokenizer corresponding to my embeddings template
bert_tokenizer = BertTokenizer.from_pretrained(embeddingsModel, do_lower_case=True)
# Import the embeedings from the cache (or the net if we do not have them already)

bert_model = BertModel.from_pretrained(embeddingsModel)
bert_model = bert_model.to(device)



class TextDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class TextMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        # print("From init self fields\n")
        # print(self.fields)
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    # def process(self, batch, device=None):

    #     """Convert outputs of preprocess into Tensors.

    #     Args:
    #         batch (List[List[List[str]]]): A list of length batch size.
    #             Each element is a list of the preprocess results for each
    #             field (which are lists of str "words" or feature tags.
    #         device (torch.device or str): The device on which the tensor(s)
    #             are built.

    #     Returns:
    #         torch.LongTensor or Tuple[LongTensor, LongTensor]:
    #             A tensor of shape ``(seq_len, batch_size, len(self.fields))``
    #             where the field features are ordered like ``self.fields``.
    #             If the base field returns lengths, these are also returned
    #             and have shape ``(batch_size,)``.
    #     """

    #     # batch (list(list(list))): batch_size x len(self.fields) x seq_len
    #     #***********************************
    #     print("\n \nFrom process with batch\n")
    #     print("\n*******************batch*******************\n")
    #     print(batch)


    #     print("\n\n*******************batch_by_feat*******************\n")
    #     batch_by_feat = list(zip(*batch))
    #     print(len(batch_by_feat))

    #     base_data = self.base_field.process(batch_by_feat[0], device=device)
    #     print("\n\n*******************base_data*******************\n")
    #     print(base_data)

    #     if self.base_field.include_lengths:
    #         # lengths: batch_size
    #         base_data, lengths = base_data

    #     feats = [ff.process(batch_by_feat[i], device=device)
    #              for i, (_, ff) in enumerate(self.fields[1:], 1)]
    #     print("\n\nf*******************feats*******************\n")
    #     print(feats)
    #     print("\n")
    #     levels = [base_data] + feats
    #     # data: seq_len x batch_size x len(self.fields)
    #     print("\n*******************levels*******************\n")
    #     print(levels)
    #     data = torch.stack(levels, 2)


    #     print("\n*******************data*******************\n")    #here
    #     print(data)  #here
    #     print("\n *******************Data shape*******************\n")
    #     print(data.shape)  #here
    #     print("\n*******************lengths*******************\n")
    #     print(lengths)
    #     assert False
    #     if self.base_field.include_lengths:
    #         return data, lengths
    #     else:
    #         return data

    ##**********************Numerricalise*******************************
    def numericalize(self, tokens):
    	
    	ids = []
    	#print("=======================Tokens================================\n")
    	#print(tokens)
    	for tok in tokens:
    		ids.append(bert_tokenizer.convert_tokens_to_ids(tok))
    		    	
    	# print("==================================IDs=======================\n")
    	# print(ids)

    	return ids

    #**********************Paddings and Retrun lenghts*******************************
    def pad(self, batch):
    	
    	max_len = max(len(x) for x in  batch)
    	padded, lengths = [], []


    	for x in  batch:
    		padded.append(x + [0] * max(0, max_len - len(x)))
    		lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

    	
    	# print("Paddings and Lengths\n")
    	# print("padded: ", padded, "lengths : ", lengths)
    	
    	return (padded, lengths)


    #**********************Creatig Tensors*******************************

    def to_tensors(self, padded, lengths, device):

    	
    	lengths = torch.tensor(lengths, dtype = torch.long, device = device)
    	tensor = torch.tensor(padded, dtype = torch.long, device = device)
    	tensor = tensor.contiguous()
    	tensor = tensor.t()
    	
    	# print("Tensors and lenghts\n")
    	# print("tensor: ", tensor.size(), "lengths : ", lengths.size())
    	

    	return tensor, lengths



    def bertify(self, tensor):
    	bert_model.eval()

    	bert_embeddings = bert_model(tensor, output_all_encoded_layers = False)
    	

    	# print("bert embeddings \n ")
    	# print(bert_embeddings[0])

    	return bert_embeddings[0]






    def process(self, batch, device=None):

        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """



        #batch (list(list(list))): batch_size x len(self.fields) x seq_len
        #***********************************
        # print("\n \nFrom process with batch\n")
        # print("\n*******************batch*******************\n")
        # print(batch)


        batch_by_feat = list(zip(*batch))

        #To List of List  lsit[list[]]
        batch_by_feat_bert  = [i[0] for i in batch]

        #print(batch_by_feat_bert)

        


        if self.fields[0][0] == 'src':

        	ids = self.numericalize(batch_by_feat_bert)
        	
        	#Numericalise  Tokens
        	padded, lengths = self.pad(ids)

        	#Converts Ids To Tensors
        	tensor , lengths = self.to_tensors(padded, lengths, device)

        	#Get Ebrt Embeddings        	
        	bert_embeddings = self.bertify(tensor)

        	

        	return bert_embeddings, lengths


              

        else:        	
        	base_data = self.base_field.process(batch_by_feat[0], device=device)
        	if self.base_field.include_lengths:
        		        # lengths: batch_size
        	        	base_data, lengths = base_data

	        feats = [ff.process(batch_by_feat[i], device=device)
	                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
	        levels = [base_data] + feats
	        # data: seq_len x batch_size x len(self.fields)
	        data = torch.stack(levels, 2)


	        if self.base_field.include_lengths:
	        	 return data, lengths
	        else:
	            return data

        

    # def preprocess(self, x):
    #     """Preprocess data.

    #     Args:
    #         x (str): A sentence string (words joined by whitespace).

    #     Returns:
    #         List[List[str]]: A list of length ``len(self.fields)`` containing
    #             lists of tokens/feature tags for the sentence. The output
    #             is ordered like ``self.fields``.
    #     """

    #     ##########################
      
    #     print("-----------------------From Preprocess---------------------\n-")
    #     print("-----------------------X---------------------\n-")
    #     print(x)
    #     print("-----------------------[f.preprocess(x) for _, f in self.fields]---------------------'n-")
    #     #var = [f.preprocess(x) for _, f in self.fields]

    #     var = [f.preprocess(x) for _, f in self.fields]

    #     print(var)
    #     ##########################

    #     return [f.preprocess(x) for _, f in self.fields]

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        ##########################
        # print("from or _, f in self.fields\n")
        # for _, f in self.fields:
        # 	print(_)
        # 	print("f-----------\n")
        # 	print(f)

        # 	print("f.preprocess(x)\n")
        # 	z = f
        # 	print([f.preprocess(z)])
        # 	print("\n")
  #       embeddingsModel = "bert-base-uncased"
		# #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		# # Make a tokenizer corresponding to my embeddings template
		# bert_tokenizer = BertTokenizer.from_pretrained(embeddingsModel)
		# # Import the embeedings from the cache (or the net if we do not have them already)
		# bert_model = BertModel.from_pretrained(embeddingsModel)
		#bert_model = bert_model.to(device)
        # print("-----------------------From Preprocess---------------------\n-")
        # print("-----------------------X---------------------\n-")
        # print(x)
        # #print("-----------------------[f.preprocess(x) for _, f in self.fields]---------------------'n-")
        # #var = [f.preprocess(x) for _, f in self.fields]
        # print("-----------------------bertify---------------------'n-")

        # var = [bertify(x)]
        # print(var)
        ##########################
        # bert_tokens = bert_tokenizer(x)
        # return [bert_tokens]
        # rint(len(self.fields))
        #print(x)
        

        #x = [i.lower() for i in x]
        #print('>>>>>>>>>>>>>>>>\n',x)
        

        if self.fields[0][0] == 'src':


        	bert_tokens = bert_tokenizer.tokenize(x.lower())
        	bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]



        	return [bert_tokens]


        else:   

        	return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]



def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
