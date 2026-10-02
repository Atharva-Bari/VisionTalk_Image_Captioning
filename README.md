# VisionTalk – Image Captioning using CNN + LSTM

Internship project: generate a natural-language caption for an image using a pretrained CNN encoder and an LSTM language decoder.

This is a **prototype** for a Master's / internship demonstration. It is not a production assistive device, medical product, or real-time camera system.

## Project overview

VisionTalk takes an image, converts it into a numeric feature vector with a frozen ImageNet CNN, then predicts caption words one by one with an LSTM until it emits an end token.

Example (after the model has been trained on Flickr8k):

- Input: a photograph of a dog playing with a ball
- Output: a sentence such as "a dog is playing with a ball in the grass"

Quality depends on training. This README does **not** claim a BLEU score until you run evaluation yourself.

## Problem statement

Describing an image in language requires two capabilities at once:

1. Computer vision: what objects and scene layout are present
2. Natural language generation: a fluent sentence, not a bag of tags

A CNN is strong at (1). An LSTM trained with teacher forcing is a standard student-scale approach to (2).

## Objective

Build a complete, Windows-runnable pipeline:

- load Flickr8k (user-provided, not auto-downloaded)
- clean captions and build a **training-only** vocabulary
- cache CNN features on disk
- train a two-input Keras model (image features + partial caption)
- generate captions (greedy decoding; optional beam search)
- report BLEU-1 and BLEU-4 on the **test** split
- expose a Streamlit demo for a single uploaded image

## Real-world application (concept vs this prototype)

**Concept:** a visually impaired user could point a camera at a scene; a captioning model produces text that a screen reader or text-to-speech engine could speak.

**This repository:** a research prototype that captions a still image on a student PC. It does not:

- run on a phone camera in real time
- guarantee safety-critical accuracy
- replace a human assistant or a certified accessibility product

Treat demo captions as research output, not ground truth.

## Architecture (viva-friendly)

```
IMAGE
  → pretrained CNN encoder (EfficientNetB0, classifier head removed)
  → image feature vector (length 1280)
  → Dense + Dropout          (image branch)

PARTIAL CAPTION
  → word IDs (padded)
  → Embedding
  → LSTM                     (caption branch)

CONCATENATE image branch + LSTM output
  → Dense
  → Softmax over vocabulary
  → next word
  → append word, repeat until endseq or max length
```

Training uses **teacher forcing**: the model sees the true prefix `startseq a dog` and must predict `is`, then `running`, then `endseq`, rather than its own previous guesses.

Inference uses the model's own predictions (greedy argmax by default).

## Technologies

- Python 3.10+ recommended
- TensorFlow / `tf.keras` (Functional API)
- NumPy, Pandas, Pillow, scikit-learn
- NLTK (BLEU)
- Matplotlib (loss curve and test-image grid)
- Streamlit (demo UI)
- tqdm (progress where used)

Use a current TensorFlow 2.x build that matches your Python version. CPU is enough for the pipeline; a GPU is used automatically if TensorFlow sees one. CUDA is **not** required.

## Dataset

**Flickr8k** (and later you may point the same code at Flickr30k by changing paths in `src/config.py`).

The project **does not download** the dataset.

Place files under `data/raw/` using one of these layouts.

### Layout A (original Flickr8k text package)

```
data/raw/Flickr8k_Dataset/          # .jpg files (a nested folder is also detected)
data/raw/Flickr8k_text/Flickr8k.token.txt
data/raw/Flickr8k_text/Flickr_8k.trainImages.txt   # optional official split
data/raw/Flickr8k_text/Flickr_8k.devImages.txt
data/raw/Flickr8k_text/Flickr_8k.testImages.txt
```

### Layout B (common Kaggle zip)

```
data/raw/Flickr8k/Images/           # .jpg files
data/raw/Flickr8k/captions.txt      # CSV with image,caption columns
```

If neither layout is present you will see:

`Flickr8k dataset not found. Place the dataset in data/raw/ according to README.md and rerun preprocessing.`

Override paths in `src/config.py`: `DATASET_PATH`, `CAPTION_FILE`, `FEATURES_VECTOR_DIR`, `MODEL_PATH`, `CHECKPOINT_PATH`.

## Project structure

```
VisionTalk_Image_Captioning/
├── data/raw/                 # Flickr8k (you provide)
├── data/processed/           # captions, tokenizer, splits
├── data/features/vectors/    # one .npy CNN vector per image
├── models/checkpoints/       # latest + best .keras files
├── models/final/             # exported model
├── notebooks/
├── outputs/figures/
├── outputs/evaluation/
├── outputs/predictions/
├── src/                      # library + CLI modules
├── app/streamlit_app.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Installation (Windows PowerShell)

```powershell
cd VisionTalk_Image_Captioning
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Installation (Windows CMD / Command Prompt)

```cmd
cd VisionTalk_Image_Captioning
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Important:** Do **not** type the markdown backticks `` ` `` shown in examples. Type only the command inside them.

If TensorFlow's pip wheel does not install, check that your Python version is supported on [tensorflow.org](https://www.tensorflow.org/install) and create a matching venv.

## Dataset setup

1. Obtain Flickr8k from its official / academic source.
2. Copy images and caption files into `data/raw/` as shown above.
3. Confirm with:

```powershell
python main.py status
```

## Preprocessing

Lowercase, strip punctuation, drop empty captions, wrap with `startseq` / `endseq`, split **by image** (no caption leakage), build the vocabulary from **training captions only**, save tokenizer and sequences.

```powershell
python -m src.text_preprocessing
```

or `python main.py preprocess`

Re-run from scratch: `python -m src.text_preprocessing --force`

## Feature extraction

Frozen EfficientNetB0 (`include_top=False`, global average pooling). Features are written as `data/features/vectors/<image>.npy`. Existing files are skipped so interrupted runs resume. The CNN is **not** run again during each training epoch.

Smoke test on a handful of images:

```powershell
python -m src.feature_extraction --limit 8
```

Full dataset (slow; run once):

```powershell
python -m src.feature_extraction
```

## Training

Default `EPOCHS = 3` in `src/config.py` is only a pipeline smoke test. After the pipeline works, increase `EPOCHS` to **20–50**.

```powershell
python -m src.train
```

Resume from `models/checkpoints/latest_caption_model.keras`:

```powershell
python -m src.train --resume
```

Callbacks: best checkpoint (never replaced by a worse `val_loss`), latest checkpoint every epoch, early stopping, reduce learning rate on plateau. Existing checkpoints are not deleted.

## Inference

Encodes **one** image. Does not preprocess Flickr8k or rebuild the tokenizer.

```powershell
python -m src.inference --image "path\to\image.jpg"
```

Optional beam search: `--beam --beam-size 3`

## Evaluation

BLEU-1 and BLEU-4 on the **test** split only (`startseq` / `endseq` are stripped). Results go to `outputs/evaluation/`. A 10-image grid is saved under `outputs/figures/`. Predictions go to `outputs/predictions/`.

```powershell
python -m src.evaluation
```

Do not quote a BLEU number in a report until this command has actually finished on your trained weights.

## Streamlit application

```powershell
streamlit run app/streamlit_app.py
```

Upload an image, preview it, click **Generate Caption**. If the tokenizer or trained model is missing, the UI tells you which command to run. The app does not train, download Flickr8k, or extract the full feature cache.

## Example commands (copy/paste)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py status
python -m src.text_preprocessing
python -m src.feature_extraction --limit 8
python -m src.feature_extraction
python -m src.train
python -m src.inference --image "data\raw\Flickr8k_Dataset\example.jpg"
python -m src.evaluation
streamlit run app/streamlit_app.py
```

Lightweight tests (no full Flickr8k required except the optional CNN cache test):

```powershell
python tests\test_preprocessing.py
python tests\test_model.py
python tests\test_inference.py
python tests\test_feature_extraction.py
```

The last command downloads ImageNet EfficientNetB0 weights on first run and uses a few synthetic JPEGs (not the 8k corpus).

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Dataset not found | Copy Flickr8k into `data/raw/` as in Dataset setup. Do not expect an automatic download. |
| Many missing images | Filenames in the caption file must match files on disk. Adjust `src/config.py`. |
| Tokenizer not found | `python -m src.text_preprocessing` |
| Missing CNN features | `python -m src.feature_extraction` |
| Missing trained model | `python -m src.train` |
| Feature dim mismatch | Re-extract features and retrain after changing `CNN_MODEL_NAME` / `IMAGE_SIZE`. |
| TensorFlow install fails | Use a Python version with an official TF wheel; CPU wheels are enough. |
| Out of memory | Lower `BATCH_SIZE` and `FEATURE_BATCH_SIZE` in `src/config.py`. |
| Captions are nonsense | Expected with 3 epochs. Train 20–50 epochs on the full feature cache. |

## Future improvements (not in this prototype)

- Attention (Show, Attend and Tell)
- Transformer / ViT / BLIP-style captioning
- Default beam search in the UI
- Flickr30k and COCO
- Multilingual captions
- Text-to-speech for an assistive demo
- Mobile / quantized deployment
- Saliency or attention maps for explainability

The assessed architecture remains **CNN + LSTM**.

## Limitations

- Flickr8k is small and dated; captions are short English sentences.
- Greedy decoding can be repetitive.
- No attention: the LSTM sees one global image vector.
- BLEU is a n-gram overlap metric, not a full measure of usefulness or safety.
- Not evaluated as an accessibility product.

## Configuration

All important knobs live in `src/config.py`:

`DATASET_PATH`, `CAPTION_FILE`, `FEATURES_VECTOR_DIR`, `MODEL_PATH`, `CHECKPOINT_PATH`, `CNN_MODEL_NAME`, `IMAGE_SIZE`, `EMBEDDING_DIM`, `LSTM_UNITS`, `DENSE_UNITS`, `DROPOUT`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `MAX_VOCAB_SIZE`, `RANDOM_SEED`.
