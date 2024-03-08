# iScore LLM server

To process summaries written in the iScore interface, we use the Hugging Face Transformers API to download transformer models and run a Python Flask server that recieves summaries from the interface, runs the models, and sends the processed data back to the interface.

The first time you run `main.py` you will download NLTK packages and the Hugging Face Transformers models. This may take a while depending on your internet download speed.

- The NLTK packages will take up around 100MB of space.
- The Hugging Face transformers will take up around 4GB of space.
  - `tiedaar/longformer-content-global` is ~0.6GB.
  - `tiedaar/longformer-wording-global` is ~0.6GB.
  - `tiedaar/summary-longformer-content` is ~0.6GB.
  - `tiedaar/summary-longformer-wording` is ~0.6GB.
  - `bloomberg/KeyBART` is ~1.5GB.

Please install Python `v3.9.x` on your computer first ([latest release](https://www.python.org/downloads/release/python-3913/))

## Setup

1. Open a command-line shell (Windows) or Terminal (MacOS, Linux) in a new window
2. Navigate to this folder (`server/`)

Windows:

3. Run `py -3.9 -m venv venv`
4. Run `.\venv\Scripts\activate`

MacOS / Linux:

3. Run `python3.9 -m venv venv`
4. Run `source venv/bin/activate`

Both:

5. Run `python -m pip install -r requirements.txt`
6. Install symspellpy `v6.7.7` ([instructions](https://symspellpy.readthedocs.io/en/latest/users/installing.html))
   - symspellpy is a Python port of SymSpell `v6.7.1`
   - **Warning for MacOS users**
     - `NOTE: symspellpy has only been tested on Windows and Linux systems and is assumed to work on macOS.`
7. Install PyTorch `v2.0.x` ([instructions](https://pytorch.org/get-started/locally/))
   - PyTorch is installed separately because some systems may support CUDA, which requires a different installation process and can significantly speed up the tool.
   - First, check if your GPU can support CUDA ([link](https://developer.nvidia.com/cuda-gpus))
   - Then, follow the instructions linked above to determine if your system can support CUDA for computation.
8. Run `python main.py`

## Packages

- Flask `v2.3.x`
- Flask-CORS `v3.0.x`
- NLTK `v3.8.x`
- NumPy `v1.24.x`
- orjson `v3.8.x`
- pandas `v2.0.x`
- PyTorch `v2.0.x` with CUDA `v11.8`
- symspellpy `v6.7.7`, a Python port of SymSpell `v6.7.1`
- Transformers `v2.1.x`
