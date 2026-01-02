# EviBiCO：Interpretable Personality Recognition via Evidence-Guided Bidirectional Collaborative Optimization

This is the repository of our paper "EviBiCO：Interpretable Personality Recognition via Evidence-Guided Bidirectional Collaborative Optimization".

## Project Structure

```
Graph-model/
├── main.py                 # Training entry point
├── adaptation.py           # Test-time adaptation with LLM Results
├── modules/
│   ├── __init__.py         # Package exports
│   ├── config.py           # Model path configuration
│   ├── dataset.py          # BigFiveDataset (PyG Data)
│   ├── gcn.py              # GCN layer
│   ├── models.py           # GCNBlock, BigFiveGNN
│   ├── losses.py           # BigFiveLoss, EvidenceWeightLoss
│   ├── evaluator.py        # ModelEvaluator
│   ├── trainer.py          # Training and evaluation functions
│   └── utils.py            # Utility functions
└── data/
    ├── BigFive_definition.py      # Big Five trait definitions
    └── multilabel_dataset.json    # Dataset(PersonalityEvd)
```

## Requirements

```
torch
torch_geometric
transformers
scikit-learn
numpy
modelscope
```

## Usage

### Training

```bash
python main.py
```

### Adaptation with LLM Results

```bash
python adaptation.py --model_path best_bigfive.pth --llm_results <llm_results.json>(You can refer to the methods in the paper to obtain it.)
```
