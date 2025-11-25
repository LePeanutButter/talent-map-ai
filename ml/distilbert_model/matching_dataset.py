import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class MatchingDataset(Dataset):
    """
    Dataset for fine-tuning models on text pair matching tasks.

    Each item in the dataset is a tuple consisting of a job description, a resume,
    and a label indicating whether they match.

    Attributes:
        items (List[Tuple[str, str, int]]): List of tuples with (job_text, resume_text, label)
        tokenizer (BertTokenizer): Tokenizer used to convert text into model-ready tokens
        max_length (int): Maximum token length for each text sequence. Longer sequences are truncated.

    Labels:
        1 -> Matching (job and resume are compatible)
        0 -> Non-matching
    """

    def __init__(self, items: List[Tuple[str, str, int]], tokenizer: DistilBertTokenizer, max_length: int = 128):
        """
        Initialize the MatchingDataset.

        Args:
            items: A list of tuples (job_text, resume_text, label) where:
                - job_text: string containing the job description
                - resume_text: string containing the candidate's resume
                - label: integer, 1 if matching, 0 if not
            tokenizer: HuggingFace BERT tokenizer to tokenize text inputs
            max_length: Maximum number of tokens per text (default 128)
        """
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieve a dataset item by index.

        Tokenizes the job and resume texts into BERT input format, truncates and pads
        them to max_length, and converts the label to a float tensor.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
                - job_tokens: tokenized job text (input_ids, attention_mask, token_type_ids)
                - res_tokens: tokenized resume text (input_ids, attention_mask, token_type_ids)
                - label: tensor containing 1.0 if matching, 0.0 if not
        """
        job_text, resume_text, label = self.items[idx]

        job_tokens = self.tokenizer(
            job_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        res_tokens = self.tokenizer(
            resume_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        job_tokens = {k: v.squeeze(0) for k, v in job_tokens.items()}
        res_tokens = {k: v.squeeze(0) for k, v in res_tokens.items()}

        return job_tokens, res_tokens, torch.tensor(label, dtype=torch.float)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for dynamic padding.
        Pads sequences to the longest in the batch, not to max_length.

        Usage:
            DataLoader(dataset, batch_size=32, collate_fn=OptimizedMatchingDataset.collate_fn)
        """
        job_tokens_list = []
        res_tokens_list = []
        labels = []

        for job_tokens, res_tokens, label in batch:
            job_tokens_list.append(job_tokens)
            res_tokens_list.append(res_tokens)
            labels.append(label)

        def pad_sequences(token_dicts):
            keys = token_dicts[0].keys()
            max_len = max(t['input_ids'].size(0) for t in token_dicts)

            padded = {key: [] for key in keys}
            for tokens in token_dicts:
                for key in keys:
                    seq = tokens[key]
                    pad_size = max_len - seq.size(0)
                    if pad_size > 0:
                        pad_value = 0
                        padded_seq = torch.cat([seq, torch.full((pad_size,), pad_value, dtype=seq.dtype)])
                    else:
                        padded_seq = seq
                    padded[key].append(padded_seq)

            return {key: torch.stack(val) for key, val in padded.items()}

        job_batch = pad_sequences(job_tokens_list)
        res_batch = pad_sequences(res_tokens_list)
        labels_batch = torch.stack(labels)

        return job_batch, res_batch, labels_batch