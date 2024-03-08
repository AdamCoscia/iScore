# iScore

[![license](https://img.shields.io/badge/License-MIT-A54046)](https://github.com/AdamCoscia/iScore/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2403.04760-red)](https://arxiv.org/abs/2403.04760)
[![DOI:10.1145/3640543.3645142](https://img.shields.io/badge/DOI-10.1145/3640543.3645142-blue)](https://doi.org/10.1145/3640543.3645142)

Upload, score, and visually compare multiple LLM-graded summaries simultaneously!

✍️📜🧑‍🏫📖💯

![The iScore System](https://github.com/AdamCoscia/iScore/blob/main/images/iscore.png)

## What is iScore?

The popularity of large language models (LLMs) has inspired LLM developers to incorporate them into adaptive educational tools that can **automatically score a summary written on a larger body of text** in a variety of educational settings such as classrooms and textbooks.
Interactively exploring how LLMs score different summaries can help developers understand the decisions on which the LLMs base their scores, discover unintended biases, update the LLMs to address the biases and mitigate the potential pedagogical ramifications of prematurely deploying untested LLM-powered educational technologies.

iScore is an interactive visual analytics tool for developers to upload, score, and compare multiple summaries of a source text simultaneously.
iScore introduces a new workflow for comparing the language features that contribute to different LLM scores:

1. First, users **upload, score and can manually revise and re-score** multiple source/summary pairs simultaneously.
2. Then, users can **visually track how scores change** across revisions in the context of expert-scored LLM training data.
3. Finally, users can **compare model weights** between words across model layers, as well as differences in scores between automatically revised **summary perturbations**.

Together, the views provide LLM developers with access to multiple summary comparison visualizations and several well-known LLM interpretability methods including attention attribution, input perturbation, and adversarial examples.
Combining these visualizations and methods in a single visual interface broadly enables deeper analysis of LLM behavior that was previously **time-consuming** and **difficult to perform**.

This code accompanies the research paper:

**[iScore: Visual Analytics for Interpreting How Language Models Automatically Score Summaries][paper]**  
<span style="opacity: 70%">Adam Coscia, Langdon Holmes, Wesley Morris, Joon Suh Choi, Scott Crossley, Alex Endert</span>  
_ACM Conference on Intelligent User Interfaces (IUI), 2024_  
| [📖 Paper][paper] | [▶️ Live Demo][demo] | [🎞️ Demo Video][video] | [🧑‍💻 Code][code] |

## Features

<details>
  <summary> 🔍 Track how scores change across revisions, in the context of expert-scored training data:</summary>
  <img src="https://github.com/AdamCoscia/iScore/blob/main/images/scores-dashboard.png" width="60%">
</details>

<details>
  <summary> ✏️ Compare differences in scores between automatically revised summary perturbations:</summary>
  <img src="https://github.com/AdamCoscia/iScore/blob/main/images/input-perturbation.png" width="60%">
</details>

<details>
  <summary> 📊 Analyze model weights between tokens across model layers:</summary>
  <img src="https://github.com/AdamCoscia/iScore/blob/main/images/token-attention.png" width="60%">
</details>

### Demo Video

🎞️ Watch the demo video for a full tutorial here: <https://youtu.be/EYJX-_fQPf0>

## Live Demo

🚀 For a live demo, visit: <https://adamcoscia.com/papers/iscore/demo/>

## Getting Started

🌱 You can test our visualizations on your own LLMs in just a few easy steps!

- Install Python `v3.9.x` ([latest release](https://www.python.org/downloads/release/python-3913/))
- Clone this repo to your computer ([instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))

```bash
git clone git@github.com:AdamCoscia/iScore.git

# use --depth if you don't want to download the whole commit history
git clone --depth 1 git@github.com:AdamCoscia/iScore.git
```

### Interface

- A frontend vanilla HTML/CSS/JavaScript web app with D3.js and Tabulator!
- Additional details can be found in [interface/README.md](interface/README.md)

Navigate to the interface folder:

```bash
cd interface
```

- If you are running Windows:

```bash
py -3.9 -m http.server
```

- If you are running MacOS / Linux:

```bash
python3.9 -m http.server
```

Navigate to [localhost:8000](http://localhost:8000/). You should see iScore running in your browser :)

### Server

- A backend Python 3.9 Flask web app to run local LLM models downloaded from Hugging Face!
- Additional details can be found in [server/README.md](server/README.md)

Navigate to the server folder:

```bash
cd server
```

Create a virtual environment:

- If you are running Windows:

```bash
# Start a virtual environment
py -3.9 -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

- If you are running MacOS / Linux:

```bash
# Start a virtual environment
python3.9 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install symspellpy `v6.7.7` ([instructions](https://symspellpy.readthedocs.io/en/latest/users/installing.html))

> symspellpy is a Python port of SymSpell `v6.7.1`  
> **Warning for MacOS users** - symspellpy has only been tested on Windows and Linux systems and is assumed to work on macOS!

Install PyTorch `v2.0.x` ([instructions](https://pytorch.org/get-started/locally/))

> PyTorch is installed separately because some systems may support CUDA, which requires a different installation process and can significantly speed up the tool.

1. First, check if your GPU can support CUDA ([link](https://developer.nvidia.com/cuda-gpus))
2. Then, follow the instructions linked above to determine if your system can support CUDA for computation.

Then run the server:

```bash
python main.py
```

## Credits

Led by <a href='https://adamcoscia.com/' target='_blank'>Adam Coscia</a>, iScore is a result of a collaboration between visualization experts in human centered computing and interaction design as well as learning engineers with expertise in natural language processing (NLP) and developing learning tools from
<img src="https://adamcoscia.com/assets/icons/other/gt-logo.png" alt="Interlocking GT" height="21" style="vertical-align: bottom;"/>
Georgia Tech,
<img src="https://adamcoscia.com/assets/icons/other/vanderbilt-logo.svg" alt="Interlocking GT" height="21" style="vertical-align: bottom;"/>
Vanderbilt, and
<img src="https://adamcoscia.com/assets/icons/other/gsu-logo.jpg" alt="Interlocking GT" height="21" style="vertical-align: bottom;"/>
Georgia State.

iScore is created by
<a href='https://adamcoscia.com/' target='_blank'>Adam Coscia</a>,
Langdon Holmes,
Wesley Morris,
Joon Suh Choi,
Scott Crossley,
and
Alex Endert.

## Citation

To learn more about iScore, please read our [research paper][paper] (published at [IUI '24](https://iui.acm.org/2024/)).

```bibtex
@inproceedings{Coscia:2024:iScore,
  author = {Coscia, Adam and Holmes, Langdon and Morris, Wesley and Choi, Joon S. and Crossley, Scott and Endert, Alex},
  title = {iScore: Visual Analytics for Interpreting How Language Models Automatically Score Summaries},
  year = {2024},
  isbn = {979-8-4007-0508-3/24/03},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3640543.3645142},
  doi = {10.1145/3640543.3645142},
  booktitle = {Proceedings of the 2024 IUI Conference on Intelligent User Interfaces},
  location = {Greenville, SC, USA},
  series = {IUI '24}
}
```

## License

The software is available under the [MIT License](https://github.com/AdamCoscia/iScore/blob/main/LICENSE).

## Contact

If you have any questions, feel free to [open an issue](https://github.com/AdamCoscia/iScore/issues) or contact [Adam Coscia](https://adamcoscia.com).

[paper]: https://arxiv.org/abs/2403.04760
[demo]: https://adamcoscia.com/papers/iscore/demo/
[video]: https://youtu.be/EYJX-_fQPf0
[code]: https://github.com/AdamCoscia/iScore
