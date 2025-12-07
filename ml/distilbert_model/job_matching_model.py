import os
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
import tempfile
import zipfile
import lzma
from transformers import DistilBertTokenizer, DistilBertModel

class JobMatchingModel(nn.Module):
    """
    A neural network model for matching job descriptions with resumes using BERT embeddings.

    This model can operate in two modes:
    1. Cosine similarity mode: computes the similarity between job and resume embeddings.
    2. Classification mode: concatenates embeddings and outputs a probability of match (0-1).

    The model consists of:
        - A pretrained BERT encoder for textual representation.
        - A projection layer to reduce DistilBERT embeddings to 256 dimensions.
        - An optional classifier for supervised matching.
    """

    ERROR_INVALID_MODE = "mode must be 'cosine' or 'clf'"
    BERT_MODEL = "distilbert-base-uncased"

    def __init__(self, model_name: str = BERT_MODEL, device: str = None, freeze_bert: bool = False):
        """
        Initialize the JobMatchingModel.

        Args:
            model_name: Name of the pretrained DistilBERT model to use.
            device: Device to run the model on ('cuda' or 'cpu'). Defaults to CUDA if available.
            freeze_bert: If True, freezes BERT parameters to prevent gradient updates.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.hidden_size: int = int(self.bert.config.hidden_size)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.to(self.device)

    def encode(self, tokens: dict) -> torch.Tensor:
        """
        Encode tokenized input text into a 256-dimensional normalized embedding.

        Args:
            tokens: Dictionary containing BERT inputs (input_ids, attention_mask, token_type_ids if applicable).

        Returns:
            Tensor of shape (batch_size, 256) containing normalized embeddings.
        """
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.bert(**tokens, return_dict=True)
        mask = tokens.get("attention_mask", None)
        last_hidden = outputs.last_hidden_state
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(-1)
            summed = (last_hidden * mask).sum(1)
            denom = mask.sum(1).clamp(min=1e-9)
            pooled = summed / denom
        else:
            pooled = last_hidden.mean(dim=1)

        projected = self.projector(pooled)
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        return normalized

    def forward(self, job_tokens: dict, res_tokens: dict, mode: str = "cosine") -> torch.Tensor:
        """
        Forward pass to compute similarity or match probability between job and resume.

        Args:
            job_tokens: Tokenized job descriptions.
            res_tokens: Tokenized resumes.
            mode: 'cosine' for cosine similarity, 'clf' for classification probability.

        Returns:
            If mode='cosine': Tensor of shape (batch_size,) with cosine similarities.
            If mode='clf': Tensor of shape (batch_size,) with match probabilities (0-1).
        """
        job_emb = self.encode(job_tokens)
        res_emb = self.encode(res_tokens)

        if mode == "cosine":
            sim = (job_emb * res_emb).sum(dim=1)
            return sim
        elif mode == "clf":
            concat = torch.cat([job_emb, res_emb], dim=1)
            prob = self.classifier(concat).squeeze(1)
            return prob
        else:
            raise ValueError(JobMatchingModel.ERROR_INVALID_MODE)

    def predict(self, job_text: str, resume_text: str, mode: str = "cosine") -> float:
        """
        Predict the match score between a job description and a resume.

        Args:
            job_text: Job description text.
            resume_text: Resume text.
            mode: 'cosine' for cosine similarity, 'clf' for classification probability.

        Returns:
            Similarity score (cosine mode) or match probability (clf mode) as a float.
        """
        self.eval()
        with torch.no_grad():
            job_tokens = self.tokenizer(
                job_text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            res_tokens = self.tokenizer(
                resume_text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )

            job_tokens = {k: v.to(self.device) for k, v in job_tokens.items()}
            res_tokens = {k: v.to(self.device) for k, v in res_tokens.items()}

            score = self.forward(job_tokens, res_tokens, mode=mode)
            return score.item()

    @staticmethod
    def _compress_file(input_path: str, output_path: str, method: str = "xz"):
        """Compress file using xz or zip."""
        if method == "xz":
            with open(input_path, "rb") as f_in, lzma.open(output_path, "wb", preset=9) as f_out:
                f_out.write(f_in.read())
        elif method == "zip":
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(input_path, arcname=os.path.basename(input_path))
        else:
            raise ValueError("Invalid compression method: choose 'xz' or 'zip'")
        return output_path

    def save(self, path: str, store_local: bool = True, compression="xz"):
        """
        Save the model checkpoint in compressed form.

        Args:
            path: Base filename, WITHOUT extension (e.g. 'models/matcher_v1')
                  OR with .pt extension (will be stripped)
            store_local: True = write to disk, False = return bytes only
            compression: 'xz' (best compression) or 'zip'
        """
        if path.endswith('.pt'):
            path = path[:-3]

        base_dir = os.path.dirname(path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        raw_path = path + ".pt"
        compressed_path = raw_path + ".xz" if compression == "xz" else path + ".zip"

        checkpoint = {
            "state_dict": self.state_dict(),
            "model_name": self.tokenizer.name_or_path,
            "hidden_size": self.hidden_size,
            "freeze_bert": not any(p.requires_grad for p in self.bert.parameters()),
        }

        tokenizer_config = self.tokenizer.save_vocabulary(os.path.dirname(raw_path) or ".")
        checkpoint["tokenizer_config"] = {
            "vocab_file": tokenizer_config[0] if isinstance(tokenizer_config, tuple) else tokenizer_config,
            "model_name": self.tokenizer.name_or_path
        }

        torch.save(checkpoint, raw_path)

        final_path = JobMatchingModel._compress_file(raw_path, compressed_path, method=compression)

        if not store_local:
            with open(final_path, "rb") as f:
                data = f.read()
            os.remove(raw_path)
            os.remove(final_path)
            return data

        os.remove(raw_path)

        print(f"Compressed model saved to: {final_path}")
        return final_path

    @classmethod
    def load(cls, path_or_bytes, device: str = None):
        """
        Load compressed or raw model.
        Accepts:
            - local path (.pt, .xz, .zip)
            - bytes (from Cassandra or other DB)
        """

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(path_or_bytes, str):
            path = path_or_bytes

            if path.endswith(".pt"):
                checkpoint = torch.load(path, map_location=device)

            elif path.endswith(".pt.xz"):
                with lzma.open(path, "rb") as f:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        tmp.write(f.read())
                        raw_tmp = tmp.name
                checkpoint = torch.load(raw_tmp, map_location=device)
                os.remove(raw_tmp)

            elif path.endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zipf:
                    name = zipf.namelist()[0]
                    with zipf.open(name) as f:
                        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                            tmp.write(f.read())
                            raw_tmp = tmp.name
                checkpoint = torch.load(raw_tmp, map_location=device)
                os.remove(raw_tmp)

            else:
                raise ValueError("File must be .pt, .pt.xz or .zip")

        else:
            with tempfile.NamedTemporaryFile(suffix=".pt.xz", delete=False) as tmp:
                tmp.write(path_or_bytes)
                compressed_path = tmp.name

            return cls.load(compressed_path, device=device)

        model = cls(
            model_name=checkpoint.get("model_name", JobMatchingModel.BERT_MODEL),
            device=device,
            freeze_bert=checkpoint.get("freeze_bert", False)
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        print(f"Model loaded successfully on {device}")
        return model

    @staticmethod
    def evaluate(model: 'JobMatchingModel',
                 loader: DataLoader,
                 loss_fn: nn.Module,
                 mode: str = "cosine") -> float:
        """
        Evaluates the model on a given data loader and returns the average loss.

        Args:
            model: The JobMatchingModel instance.
            loader: DataLoader for the validation/test dataset.
            loss_fn: The loss function used for the respective mode.
            mode: 'cosine' or 'clf'.

        Returns:
            Average loss over the dataset.
        """
        model.eval()
        device = model.device
        total_loss = 0.0

        with torch.no_grad():
            for job_tokens, res_tokens, labels in loader:
                labels = labels.to(device)

                if mode == "cosine":
                    labels_ce = labels.clone()
                    labels_ce[torch.isclose(labels_ce, torch.tensor(0.0))] = -1.0

                    job_emb = model.encode({k: v.to(device) for k, v in job_tokens.items()})
                    res_emb = model.encode({k: v.to(device) for k, v in res_tokens.items()})

                    loss = loss_fn(job_emb, res_emb, labels_ce)

                elif mode == "clf":
                    preds = model(job_tokens, res_tokens, mode="clf")
                    loss = loss_fn(preds, labels)

                else:
                    raise ValueError(JobMatchingModel.ERROR_INVALID_MODE)

                total_loss += loss.item()

        model.train()
        return total_loss / len(loader)

    @staticmethod
    def train_loop(model: 'JobMatchingModel',
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   epochs: int = 3,
                   lr: float = 2e-5,
                   weight_decay: float = 1e-4,
                   mode: str = "cosine"):
        """
        The main training loop for the JobMatchingModel.

        Args:
            model: The JobMatchingModel instance.
            train_loader: DataLoader for the training data.
            val_loader: Optional DataLoader for validation data.
            epochs: Number of training epochs.
            lr: Learning rate for the optimizer.
            weight_decay: L2 penalty for the optimizer (default 1e-4).
            mode: 'cosine' (for CosineEmbeddingLoss with labels 1/-1) or 'clf' (for BCELoss).
        """
        device = model.device
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        if mode == "cosine":
            loss_fn = nn.CosineEmbeddingLoss(margin=0.25, reduction='mean')
        elif mode == "clf":
            loss_fn = nn.BCELoss(reduction='mean')
        else:
            raise ValueError(JobMatchingModel.ERROR_INVALID_MODE)

        model.train()
        print(f"Starting training for {epochs} epochs in **{mode}** mode on device: {device}")

        for epoch in range(epochs):
            running_loss = 0.0

            for job_tokens, res_tokens, labels in train_loader:
                labels = labels.to(device)
                optimizer.zero_grad()

                if mode == "cosine":
                    labels_ce = labels.clone()
                    labels_ce[torch.isclose(labels_ce, torch.tensor(0.0))] = -1.0
                    job_emb = model.encode({k: v.to(device) for k, v in job_tokens.items()})
                    res_emb = model.encode({k: v.to(device) for k, v in res_tokens.items()})

                    loss = loss_fn(job_emb, res_emb, labels_ce)

                else:
                    preds = model(job_tokens, res_tokens, mode="clf")
                    loss = loss_fn(preds, labels.to(device))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            if val_loader is not None:
                eval_loss = JobMatchingModel.evaluate(model, val_loader, loss_fn, mode=mode)
                print(f"Val Loss: {eval_loss:.4f}")