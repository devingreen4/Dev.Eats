import torch
import clip
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import random

class FoodAnalyzer():
    def __init__(self, 
                 # I have a Mac, so I accelerate torch using MPS
                 device="mps", 
                 model="ViT-B/32", 
                 classes="classes",
                 templates="templates",
                 responses="responses"):
        
        self.device = device
        self.model, self.preprocess = clip.load(model, device=device)
        self.classes = self.__open(classes)
        self.templates = self.__open(templates)
        self.responses = self.__open(responses)

        self.features = self.__generate_features()
        print("Food Analyzer Initialized!")

    def __open(self, filename):
        with open(f"./data/{filename}.txt") as file:
            return [ line.strip() for line in file ]

    def __generate_features(self) -> torch.Tensor:
        all_features = []

        with torch.no_grad():
            for class_name in tqdm(self.classes, desc="Encoding prompts"):
                texts = [ template.format(class_name) for template in self.templates ]
                tokens = clip.tokenize(texts).to(self.device)

                class_features = self.model.encode_text(tokens)
                class_features /= class_features.norm(dim=-1, keepdim=True)

                mean_features = class_features.mean(dim=0)
                mean_features /= mean_features.norm()

                all_features.append(mean_features)

        return torch.stack(all_features, dim=0)

    def predict(self, image: Image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ self.features.T).softmax(dim=-1)

            _, pred = torch.topk(similarity, k=1, dim=-1)
            pred = pred.squeeze(1).cpu()
        
        pred_str = self.classes[pred.item()]
        return pred_str, random.choice(self.responses).format(pred_str)