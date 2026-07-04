# Social Media Controversiality Predictor

This project predicts how a social media post may perform across like-to-retweet ratio percentiles using an end-to-end NLP pipeline. It was built for the 31st Supercomputing Challenge and is structured as a small machine learning system: collect tweet data, transform engagement metrics into labels, train a BERT-based classifier, and expose predictions through a demo interface.

The core product question is:

> Can the text of a post help predict whether engagement will look broadly agreeable, low-signal, or potentially controversial?

## Live Demo

Open `index.html` in a browser, or publish this repository with GitHub Pages.

The GitHub Pages demo is a static, browser-only approximation of the model workflow. It does not load the TensorFlow SavedModel because GitHub Pages cannot run the Python/TensorFlow backend. Instead, it demonstrates the same system contract: text input goes through feature extraction, percentile scoring, and an interpretable engagement distribution.

## System Design

The project is organized as a layered pipeline with clear ownership boundaries.

```text
User / Demo Layer
  - Gradio Python demo for local model inference
  - Static GitHub Pages demo for portfolio review

ML Inference Layer
  - TensorFlow/Keras model
  - BERT preprocessing and encoder from TensorFlow Hub
  - Dense output layer for 10 engagement percentile bins

Training and Analysis Layer
  - Train/test split
  - Like-to-retweet ratio calculation
  - Percentile label generation
  - Accuracy, precision, recall, and visualization scripts

Data Collection Layer
  - Twitter API wrapper
  - Username-based collection scripts
  - Rate-limit handling
  - JSON dataset output
```

## Why This Design

This architecture separates collection, training, inference, and presentation so each part can be changed independently. For example, the Twitter API code can be replaced with another social platform, the BERT encoder can be swapped for another language model, and the demo layer can evolve into a hosted API without rewriting the data pipeline.

The model predicts percentile bins instead of a raw like-to-retweet number. That keeps outliers from dominating the target variable and makes the output easier to explain: a post is assigned a distribution over relative engagement categories rather than a brittle exact count.

## Data Flow

```text
Twitter API
  -> raw post JSON
  -> content, likes, retweets, replies, quote count
  -> like / retweet ratio
  -> percentile bin labels
  -> BERT text preprocessing
  -> BERT encoder embeddings
  -> dense classifier
  -> 10-bin engagement distribution
```

## Key Components

| Area | Files | Purpose |
| --- | --- | --- |
| Data collection | `twitter_scraping/twitter_api.py`, `tweet_stream.py`, `ds_from_list.py` | Fetch tweets, collect public metrics, and build the dataset. |
| Rate limiting | `twitter_scraping/rate_limit.py` | Track request windows and pause before exceeding API limits. |
| Training | `traning_and_analysis/training.py` | Build percentile labels, train the BERT classifier, and save the model. |
| Local inference | `traning_and_analysis/demo.py` | Run a Gradio interface backed by the trained TensorFlow model. |
| Analysis | `traning_and_analysis/graph_maker.py`, `history_graph.py`, `percentile_accuracies.py` | Visualize model behavior and training history. |
| Static demo | `index.html` | GitHub Pages-friendly demo that explains the prediction workflow. |

## Model Architecture

```text
Tweet text
  -> BERT preprocessing layer
  -> BERT encoder
  -> pooled text embedding
  -> dropout regularization
  -> dense layer with 10 sigmoid outputs
  -> engagement percentile probabilities
```

The project uses transfer learning because training a language model from scratch would require far more data and compute. BERT provides general language understanding, while the final dense layer adapts that representation to this project's engagement prediction task.

## Engineering Tradeoffs

| Decision | Benefit | Tradeoff |
| --- | --- | --- |
| BERT transfer learning | Strong text representations with less project-specific data | Heavier dependency and slower training than classical ML |
| Percentile classification | Reduces sensitivity to viral outliers | Gives relative categories instead of exact engagement counts |
| JSON dataset | Easy to inspect and reproduce locally | Not optimized for large-scale training |
| Gradio local demo | Fast way to test real model predictions | Requires Python dependencies and a saved model |
| Static GitHub Pages demo | Easy portfolio access | Demonstrates system behavior rather than running the full model |

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python traning_and_analysis/training.py
```

Run the real local inference demo:

```bash
python traning_and_analysis/demo.py
```

Open the static portfolio demo:

```bash
start index.html
```

## What This Shows

This project demonstrates practical system design for a machine learning application:

- A modular pipeline from data collection to inference
- API abstraction and rate-limit awareness
- Feature and label design for noisy real-world engagement data
- Transfer learning with a production-style TensorFlow model
- Separate local and static demo surfaces for different deployment constraints
- Clear tradeoff thinking around scalability, explainability, and maintainability

## Future Improvements

- Save percentile boundaries alongside the trained model so inference can display calibrated labels.
- Add unit tests for API URL construction, ratio binning, and empty-text handling.
- Move configuration values such as dataset path, model path, and epochs into a config file.
- Add a small backend API for hosted real-model inference.
- Track model versions and evaluation metrics for reproducible experiments.

## Tech Stack

Python, TensorFlow, TensorFlow Hub, TensorFlow Text, BERT, NumPy, scikit-learn, Gradio, Requests, HTML, CSS, and JavaScript.

## License

See `LICENSE`.
