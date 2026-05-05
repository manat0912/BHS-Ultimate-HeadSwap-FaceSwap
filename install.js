module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/Hanzyusuf/BHS-HeadSwap.git app",
      },
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git clone https://github.com/HumanAIGC/SwapAnyHead.git"
      },
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git clone https://github.com/visomaster/VisoMaster.git"
      },
    },
    {
      method: "fs.copy",
      params: {
        src: "main.py",
        dest: "app/main.py"
      },
    },
    {
      method: "fs.copy",
      params: {
        src: "viso_bridge.py",
        dest: "app/viso_bridge.py"
      },
    },
    {
      method: "fs.copy",
      params: {
        src: "face_swappers.py",
        dest: "app/VisoMaster/app/processors/face_swappers.py"
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r ../requirements.txt"
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install tensorrt==10.6.0 tensorrt-cu12_libs==10.6.0 tensorrt-cu12_bindings==10.6.0 --extra-index-url https://pypi.nvidia.com"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "hf.download",
      params: {
        "path":"app",
        "_": [ "Alissonerdx/BFS-Best-Face-Swap" ],
        "local-dir": "BFS-Best-Face-Swap"
      }
    },
    {
      method: "hf.download",
      params: {
        "path":"app",
        "_": [ "olesheva/head_swap_qwen_edit" ]
      }
    },
    {
      method: "hf.download",
      params: {
        "_": [ "tonera/FLUX.2-klein-4B-fp8-diffusers" ]
      }
    },
    {
      method: "hf.download",
      params: {
        "_": [ "rootlocalghost/FLUX.2-klein-9B-FP8" ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app/VisoMaster",
        message: "python download_models.py"
      }
    }
  ]
}