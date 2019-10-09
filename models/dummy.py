import json
import torch

from .utils import load_model
from .modules import StrokesSynthesis, StrokesPrediction

with open("../models/alphabet.json") as fin:
    alphabet = json.load(fin)
attention_scale = 0.04551004232033393

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

prediction_model = StrokesPrediction()
load_model(prediction_model, "../models/models/prediction.pt", device=device)
prediction_model.to(device)

synthesis_model = StrokesSynthesis(num_letters=len(alphabet), attention_scale=attention_scale)
load_model(synthesis_model, "../models/models/synthesis.pt", device=device)
synthesis_model.to(device)


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    torch.manual_seed(random_seed)
    sample = prediction_model.sample(device=device)
    return sample


def generate_conditionally(text='welcome to lyrebird', bias=10, random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    torch.manual_seed(random_seed)
    text += "  "  # need some padding to know where to stop
    text_indices = torch.LongTensor([alphabet.get(x, alphabet[" "]) for x in text])[None, :].to(device)
    sample = synthesis_model.sample(text_indices, device=device, bias=bias)
    return sample


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
