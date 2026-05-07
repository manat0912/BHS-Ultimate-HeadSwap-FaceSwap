# BHS Ultimate HeadSwap & FaceSwap (Pinokio Launcher)

A 1-click installer and launcher for **BHS Ultimate HeadSwap & FaceSwap**, built for [Pinokio](https://pinokio.computer/).

## Features

- **Flux 2 Klein 4B
- **VisoMaster Fast Face Swap**: Includes blazing fast image and video swapping via the VisoMaster engine.
- **Multiple AI Models Supported**: Comes out of the box with:
  - Inswapper128
  - SimSwap512
  - GhostFace (v1, v2, v3)
  - CSCS
- **Advanced Tools Auto-Cloned**: Automatically downloads and installs dependencies from `BFS-Best-Face-Swap`, `SwapAnyHead`, `VisoMaster`.

## Installation

1. Download and install [Pinokio](https://pinokio.computer/).
2. Paste the URL of this repository into the Pinokio discover tab and click download.
3. Click **Install**. The script will automatically:
   - Create isolated virtual environments.
   - Clone all required sub-repositories and GitHub/HuggingFace assets.
   - Download the necessary ONNX and Safetensors models (via `git lfs` and direct downloads).

## Usage

1. Click **Start** to launch the Gradio Web UI.
2. The UI will start locally at `http://127.0.0.1:7860`.
3. Select your desired tool from the tabs:
   - **Flux(Head Swap)**: For photorealistic, high-quality structure and expression replication.
   - **Viso Fast Swap (Image)**: For rapid, single-image face swapping.
   - **Viso Fast Swap (Video)**: For smooth, frame-by-frame video face swapping.

## Troubleshooting

- If you encounter any issues during installation, check the **logs** folder in the Pinokio UI.
- Ensure your system meets the requirements for running PyTorch with CUDA (NVIDIA GPU recommended).

---
*Based on BFS - Best Face Swap, VisoMaster, and SwapAnyHead.*
