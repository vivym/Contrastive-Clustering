import pickle
import random
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer


class CUB200(Dataset):
    def __init__(
        self,
        root_path: str,
        split: str,
        transform = None,
    ):
        super().__init__()

        self.root_path = Path(root_path)
        self.split = split
        self.transform = transform

        with open(self.root_path / "birds" / split / "filenames.pickle", "rb") as f:
            self.file_names = pickle.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int):
        file_name = self.file_names[index]

        label = int(file_name.split(".")[0]) - 1

        image_path = self.root_path / "images" / f"{file_name}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption_path = self.root_path / "birds" / "text" / f"{file_name}.txt"
        with open(caption_path) as f:
            captions = f.readlines()
            captions = map(lambda x: x.strip(), captions)
            captions = filter(lambda x: len(x) > 0, captions)
            captions = list(captions)

        if self.split == "train":
            caption = random.choice(captions)
        else:
            caption = captions[0]

        results = self.tokenizer(
            caption,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = results["input_ids"].squeeze(0)
        token_type_ids = results["token_type_ids"].squeeze(0)
        attention_mask = results["attention_mask"].squeeze(0)

        return image, (input_ids, token_type_ids, attention_mask), label


def main():
    cub200 = CUB200(root_path="datasets/CUB_200_2011", split="train")
    cub200[0]


if __name__ == "__main__":
    main()
