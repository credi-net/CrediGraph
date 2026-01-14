<div align="center">

## CrediGraph: Graph Construction for Network-Based Credibility Modelling

<img src="img/credigraph.png" alt="CrediGraph Logo" width="100" />

### About CrediGraph

We develop billion-scale data webgraphs and use them to assess credibility levels of websites, which can be used downstream to augment Retrieval-Augmented Generation robustness and fact-checking. **CrediGraph** is the graph construction pipelines which allows us to automatically source from monthly Common Crawl data, and produce processed webgraphs usable to train credibility models and otherwise analyse trust propagation on the web. 

The **CrediBench** dataset, a benchmark of 6+ months of billion-scale webgraphs <br> constructed using the **CrediGraph** pipeline, are available on [Huggingface - CrediBench](https://huggingface.co/datasets/credi-net/CrediBench).

</div>

---

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip`, you can invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo

# Enter the repo directory
cd CrediGraph

# Install core dependencies into an isolated environment
uv sync

# The isolated env is .venv
source .venv/bin/activate
```

## Usage

### Building Temporal Graphs

The graph construction can be parallelized. We use a configuration of 16 subfolders, distributed across 16 CPUs.

To run the pipeline on a given month interval, simply run:

```sh
# First,
cd construction

# then,
bash pipeline.sh <start_month> <end_month> <number of subfolders>

# e.g,
bash pipeline.sh 'January 2020' 'February 2020' 16
```

We also have a job script ready to use, destined for usage on a cluster, which can be run with:

```bash
sbatch run.sh <start_month> [<end_month>]
```

For optimal settings, refer to `construction/README.md`.

This will construct the graph in `$SCRATCH/crawl-data/CC-MAIN-YYYY-WW/output` if a `$SCRATCH` variable is defined in the job's environment; otherwise, simply in `crawl-data/CC-MAIN-YYYY-WW/output`.

#### Processing

For processing a graph, still from `construction`, run:

```sh
bash process.sh "$START_MONTH" "$END_MONTH"
```

### Running domain's content extraction

For content extraction per month, refer to [CrediText](https://github.com/credi-net/CrediText).

### Running Experiments

For experimental results, refer to [CrediText](https://github.com/credi-net/CrediText) for content-related work, and [CrediPred](https://github.com/credi-net/CrediPred) for graph neural network experiments.

______________________________________________________________________

### Citation

```
@article{kondrupsabry2025credibench,
  title={{CrediBench: Building Web-Scale Network Datasets for Information Integrity}},
  author={Kondrup, Emma and Sabry, Sebastian and Abdallah, Hussein and Yang, Zachary and Zhou, James and Pelrine, Kellin and Godbout, Jean-Fran{\c{c}}ois and Bronstein, Michael and Rabbany, Reihaneh and Huang, Shenyang},
  journal={arXiv preprint arXiv:2509.23340},
  year={2025},
  note={New Perspectives in Graph Machine Learning Workshop @ NeurIPS 2025},
  url={https://arxiv.org/abs/2509.23340}
}
```
