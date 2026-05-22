# RSRefSeg 2: Decoupling Referring Remote Sensing Image Segmentation with Foundation Models

Deep learning framework for remote sensing image segmentation using natural language referring expressions, combining CLIP and SAM.

## Overview

Segments specific objects in satellite/aerial imagery using natural language. Uses CLIP for semantic understanding and SAM for high-quality segmentation masks via cascaded second-order prompting.

## Features

- Multi-Modal Fusion: CLIP text + image embeddings
- Cascaded Second-Order Prompter: Refined cross-modal alignment
- SAM Integration: High-quality segmentation masks
- Flexible Referencing: Diverse natural language expressions

## Architecture

1. CLIP Encoder - Text and image embeddings
2. Cascaded Second-Order Prompter - Embedding refinement
3. SAM Decoder - Segmentation mask generation

## Installation

pip install torch torchvision transformers segment-anything

## Usage



## Applications

- Urban planning
- Agricultural monitoring
- Disaster assessment
- Environmental change detection

## License

See LICENSE file.
