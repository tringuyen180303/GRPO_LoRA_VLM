from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

class CosineSimilarityCalculator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, pred: str, label: str) -> float:
        emb_pred = self.model.encode([pred], convert_to_tensor=True)
        emb_label = self.model.encode([label], convert_to_tensor=True)
        return util.cos_sim(emb_pred, emb_label).item()

class ModelEvaluator:
    def __init__(self, model_name: str, dataset_name: str, split: str = "test"):
        self.similarity_calculator = CosineSimilarityCalculator()
        self.dataset = load_dataset(dataset_name, split=split)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def evaluate(self) -> float:
        total_similarity = 0.0

        for sample in self.dataset:
            image = sample["image"]
            instruction = "You are an expert radiographer. Describe accurately what you see in this image."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            input_ids = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )

            # If your processor returns a dict on your version, keep this.
            # Otherwise, use processor(...) per the model docs.
            if isinstance(input_ids, dict):
                inputs = {k: v.to(self.model.device) for k, v in input_ids.items()}
            else:
                inputs = {"input_ids": input_ids.to(self.model.device)}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=128)

            response = self.processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            pred = response.strip()
            label = sample["caption"]

            total_similarity += self.similarity_calculator.calculate_similarity(pred, label)

        return total_similarity / len(self.dataset)

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_name = "unsloth/Radiology_mini"

    evaluator = ModelEvaluator(model_name=model_name, dataset_name=dataset_name)
    avg_similarity = evaluator.evaluate()
    print(f"Average Cosine Similarity Score: {avg_similarity:.4f}")