import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

class Predictor:
    def setup(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "PULSE-ECG/PULSE-7B",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("PULSE-ECG/PULSE-7B")

    def predict(self, image, prompt="Analyze this ECG and report abnormalities in JSON"):
        img = Image.open(image).convert("RGB")
        inputs = self.processor(prompt, img, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=512)
        result = self.processor.decode(output[0], skip_special_tokens=True)
        return {"analysis": result}
