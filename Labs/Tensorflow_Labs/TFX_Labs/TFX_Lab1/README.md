# TFX Feature Engineering Lab — Diabetes Dataset

## What is this lab about?

In this lab, I used **TensorFlow Extended (TFX)** to build a feature engineering pipeline for a medical dataset. Think of a pipeline as an assembly line — raw data goes in one end, and clean, properly formatted data comes out the other end, ready to be used for training a machine learning model.

The original lab used a breast cancer dataset. I swapped it out for the **Diabetes dataset**, which gave me a chance to handle a different kind of problem — instead of predicting yes/no (classification), this dataset predicts a number (regression), which required different preprocessing decisions.

---

## The Dataset

The dataset contains medical information for **442 diabetes patients**. Each patient has 10 features recorded at baseline, and the goal is to predict how much their diabetes progressed one year later.

| Feature | What it means |
|---------|--------------|
| `age` | Patient's age |
| `sex` | Patient's sex |
| `bmi` | Body mass index |
| `bp` | Average blood pressure |
| `s1` | Total cholesterol |
| `s2` | LDL (bad cholesterol) |
| `s3` | HDL (good cholesterol) |
| `s4` | Cholesterol/HDL ratio |
| `s5` | Triglycerides level |
| `s6` | Blood sugar level |
| `target` | Disease progression score (this is what we're predicting) |

One important thing to note — **every single feature here is a number**. There are no text or category-based columns (like "male/female" as text). This is different from the original lab and meant I had to adjust the pipeline accordingly.

---

## What I Changed from the Original Lab

The original lab was built around a breast cancer dataset that had categorical (text-based) features and a yes/no label. Here's what I changed:

| Thing | Original Lab | My Version |
|-------|-------------|------------|
| Dataset | Breast cancer | Diabetes |
| Feature types | Mix of numeric + categorical | All numeric |
| Prediction type | Classification (yes/no) | Regression (a number) |
| Label handling | Category encoding | Scale to [0,1] |
| Bucketization | None | Age and BMI split into 4 groups |

The biggest challenge was that the diabetes dataset has **no categorical features at all**, so I had to think carefully about which transformations actually make sense for this data.

---

## What the Pipeline Does

Here's what happens to the data when it goes through the pipeline:

1. **Most numeric features** (`sex`, `bp`, `s1` through `s6`) get scaled to a range between 0 and 1. This is important because machine learning models work much better when all features are on the same scale — otherwise a feature like blood pressure (which has large numbers) would unfairly dominate over a feature like a ratio (which has small numbers).

2. **Age and BMI** get split into 4 buckets each. For example, age might be split into young, middle-aged, older, and elderly groups. This can help the model pick up on patterns within age ranges rather than treating age as a purely linear number.

3. **The target label** also gets scaled to [0,1] — since we're predicting a continuous value (not a category), this keeps the output in a manageable range for training.

4. **The categorical feature loop is intentionally left empty** — there simply are no text features in this dataset, so nothing needs to be encoded.

---

## Project Structure

```
TFX_Lab1/
├── census_constants.py       # All feature names and settings defined here
├── census_transform.py       # The actual transformation logic
├── data/
│   └── census_data/
│       └── diabetes_dataset.csv   # The dataset
└── README.md
```

---

## How to Run This Lab

> **Heads up for Mac users:** TFX doesn't install cleanly on macOS (especially Apple Silicon) or Python 3.12+. The easiest and most reliable way to run this is using Docker, which sets up the right environment automatically.

### Option 1: Docker (Recommended)

**Step 1** — Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

**Step 2** — Clone this repo and navigate to the lab folder:
```bash
git clone <your-repo-url>
cd MLOps/Labs/Tensorflow_Labs/TFX_Labs/TFX_Lab1
```

**Step 3** — Run the script inside the official TFX Docker container:
```bash
docker run -it --rm \
  -v $(pwd):/home/mlops \
  -w /home/mlops \
  tensorflow/tfx:1.15.0 \
  python census_transform.py
```

This command mounts your local folder into the container and runs the script. Docker will download the TFX image the first time (~2.5GB), but it's cached after that.

---

### Option 2: Python Virtual Environment (Linux only)

**Step 1** — Create a Python 3.10 virtual environment (3.10 is required — 3.11+ won't work):
```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
```

**Step 2** — Install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install tensorflow-transform==1.15.0
```

**Step 3** — Run the script:
```bash
python census_transform.py
```

---

## Common Errors and Fixes

| Error message | What it means | How to fix it |
|---------------|--------------|---------------|
| `ModuleNotFoundError: tensorflow_transform` | The virtual environment isn't activated | Run `source .venv/bin/activate` first, or use Docker |
| `metadata-generation-failed` | Python version is too new (3.12 or 3.13) | Switch to Python 3.10 or use Docker |
| `No matching distribution for tfx-bsl` | TFX doesn't support macOS ARM natively | Use Docker |

---

## Dependencies

- `tensorflow >= 2.13`
- `tensorflow-transform >= 1.13`
- `apache-beam`
- `tfx-bsl`

If you're using Docker, you don't need to install any of these manually — they all come pre-installed in the `tensorflow/tfx:1.15.0` image.

---

## References

- [TensorFlow Transform Docs](https://www.tensorflow.org/tfx/transform/get_started)
- [Scikit-learn Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [TFX Guide](https://www.tensorflow.org/tfx/guide)
