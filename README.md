![12.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/eIE56Qk5iKuDoZ3rgkKx2.png)

# **Human-vs-NonHuman-Detection**  

> **Human-vs-NonHuman-Detection** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images as either human or non-human using the **SiglipForImageClassification** architecture.  

```py
Classification Report:
              precision    recall  f1-score   support

     Human 𖨆     0.9939    0.9735    0.9836      6646
 Non Human メ    0.9807    0.9956    0.9881      8989

    accuracy                         0.9862     15635
   macro avg     0.9873    0.9845    0.9858     15635
weighted avg     0.9863    0.9862    0.9862     15635
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ToGf2iWUKacTCQQn9hRPD.png) 

The model categorizes images into two classes:  
- **Class 0:** "Human 𖨆"  
- **Class 1:** "Non Human メ"  

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Human-vs-NonHuman-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def human_detection(image):
    """Predicts whether the image contains a human or non-human entity."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Human 𖨆", 
        "1": "Non Human メ"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=human_detection,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Human vs Non-Human Detection",
    description="Upload an image to classify whether it contains a human or non-human entity."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```  

# **Intended Use:**  

The **Human-vs-NonHuman-Detection** model is designed to distinguish between human and non-human entities. Potential use cases include:  

- **Surveillance & Security:** Enhancing monitoring systems to detect human presence.  
- **Autonomous Systems:** Helping robots and AI systems identify humans.  
- **Image Filtering:** Automatically categorizing human vs. non-human images.  
- **Smart Access Control:** Identifying human presence for secure authentication.
