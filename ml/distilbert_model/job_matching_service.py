import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from .job_matching_model import JobMatchingModel
from .matching_dataset import MatchingDataset

class JobMatchingService:
    def __init__(self, save_dir: str = None, model_name: str = "distilbert-base-uncased", device: str = None):
        """
        :param save_dir: Directory where the model will be saved.
                         If not provided, it saves in the /models/ directory.
        :param model_name: Name of the pretrained BERT model.
                           'distilbert-base-uncased' is recommended (66M params, 2x faster than BERT)
        :param device: Device to run the model on ('cuda' or 'cpu'). Defaults to CUDA if available.
        """
        if save_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            save_dir = os.path.join(project_root, "models")

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None

    def train_and_save(self,
                       model_id: str,
                       train_data: list,
                       val_data: list = None,
                       epochs: int = 5,
                       mode: str = "cosine",
                       batch_size: int = 16,
                       lr: float = 2e-5,
                       weight_decay: float = 1e-4,
                       freeze_bert: bool = True,
                       use_amp: bool = False,
                       num_workers: int = 0):
        """
        Trains a model and saves it with a unique name using model_id.

        :param model_id: Identifier for the model (e.g., name, ID).
        :param train_data: List of (job_text, resume_text, label) tuples.
        :param val_data: Optional validation data in same format.
        :param epochs: Number of epochs to train.
        :param mode: "cosine" or "clf" - determines the mode of the model.
        :param batch_size: Batch size for training.
        :param lr: Learning rate for optimizer.
        :param weight_decay: L2 regularization weight.
        :param freeze_bert: If True, only trains projection/classifier layers (much faster).
        :param use_amp: Use automatic mixed precision for faster training (requires CUDA).
        :param num_workers: Number of DataLoader workers (0 = main thread).
        :return: Path to the saved model file.
        """
        print("Training Configuration:")
        print(f"Model ID: {model_id}")
        print(f"Mode: {mode}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Freeze BERT: {freeze_bert}")
        print(f"Device: {self.device}")
        print(f"AMP enabled: {use_amp and self.device == 'cuda'}")
        print(f"Training samples: {len(train_data)}")
        if val_data:
            print(f"Validation samples: {len(val_data)}\n")

        self.model = JobMatchingModel(
            model_name=self.model_name,
            device=self.device,
            freeze_bert=freeze_bert
        )

        train_dataset = MatchingDataset(train_data, tokenizer=self.tokenizer, max_length=128)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            collate_fn=MatchingDataset.collate_fn
        )

        val_loader = None
        if val_data:
            val_dataset = MatchingDataset(val_data, tokenizer=self.tokenizer, max_length=128)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device == "cuda"),
                collate_fn=MatchingDataset.collate_fn
            )

        JobMatchingModel.train_loop(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            mode=mode
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_id}_{mode}_{timestamp}"
        save_path = os.path.join(self.save_dir, filename)

        final_path = self.model.save(save_path, compression="xz")

        print(f"Model saved to: {final_path}")

        return final_path

    def load_model(self, model_path: str) -> JobMatchingModel:
        """
        Load a saved model from disk.

        :param model_path: Path to the saved model file.
        :return: Loaded JobMatchingModel instance.
        """
        return JobMatchingModel.load(model_path, device=self.device)

    def predict(self, job_text: str, resume_text: str, mode: str = "cosine") -> float:
        """
        Make a prediction using the currently loaded model.

        :param job_text: Job description text.
        :param resume_text: Resume text.
        :param mode: 'cosine' or 'clf'.
        :return: Similarity score or match probability.
        """
        if self.model is None:
            raise ValueError("No model loaded. Train a model first or load an existing one.")

        return self.model.predict(job_text, resume_text, mode=mode)

    def batch_predict(self, job_resume_pairs: list, mode: str = "cosine", batch_size: int = 32) -> list:
        """
        Make predictions for multiple job-resume pairs efficiently.

        :param job_resume_pairs: List of (job_text, resume_text) tuples.
        :param mode: 'cosine' or 'clf'.
        :param batch_size: Batch size for inference.
        :return: List of scores.
        """
        if self.model is None:
            raise ValueError("No model loaded. Train a model first or load an existing one.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(job_resume_pairs), batch_size):
                batch = job_resume_pairs[i:i + batch_size]
                job_texts = [pair[0] for pair in batch]
                resume_texts = [pair[1] for pair in batch]

                job_tokens = self.tokenizer(
                    job_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )
                res_tokens = self.tokenizer(
                    resume_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )

                job_tokens = {k: v.to(self.device) for k, v in job_tokens.items()}
                res_tokens = {k: v.to(self.device) for k, v in res_tokens.items()}

                batch_scores = self.model.forward(job_tokens, res_tokens, mode=mode)
                scores.extend(batch_scores.cpu().tolist())

        return scores
