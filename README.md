# EthioMart Vendor Analytics & Micro-Lending Scorecard
### Project Overview
#### This project aims to:
- Scrape and preprocess Ethiopian e-commerce Telegram messages.
- Extract key business entities (products, prices, locations) using a fine-tuned NER model.
- Analyze vendor performance based on engagement metrics (views, post frequency, pricing).
- Generate a lending scorecard to help EthioMart identify high-potential vendors for micro-loans.

#### Tasks Breakdown
##### 📌 Task 1: Data Ingestion & Preprocessing
Objective: Fetch and preprocess Telegram messages for NER training.

Steps:
✅ Scrape Telegram Channels

Use telethon or pyrogram to collect messages from 5+ Ethiopian e-commerce channels.

Store raw data (text, images, metadata) in a structured format (CSV/JSON).

✅ Preprocess Text Data

Normalize Amharic text (remove emojis, special chars).

Tokenize messages for NER labeling.

Extract metadata: [vendor_name, timestamp, views].

📂 Output:

raw_data/ (raw scraped messages)

processed_data.csv (structured, cleaned data)

##### Task 2: Label Dataset in CoNLL Format
Objective: Manually annotate 30-50 messages for NER training.

Steps:
✅ Label Entities:

B-Product, I-Product (e.g., ልብስ, ሸሚዝ)

B-PRICE, I-PRICE (e.g., 1000 ብር)

B-LOC, I-LOC (e.g., Addis Ababa, Bole)

📂 Output:

ner_labels.conll (annotated dataset in CoNLL format)

##### Task 3: Fine-Tune NER Model
Objective: Train a model to extract products, prices, and locations.

Steps:
✅ Setup Environment (Google Colab GPU)

python
!pip install transformers datasets seqeval
✅ Load & Tokenize Data

Use HuggingFace Datasets to load ner_labels.conll.

Tokenize with XLM-Roberta or bert-tiny-amharic.

✅ Train Model

Fine-tune using Trainer API (5-10 epochs).

Evaluate on validation set (F1-score, precision, recall).

📂 Output:

saved_model/ (fine-tuned NER model)

eval_results.txt (performance metrics)

##### Task 4: Model Comparison & Selection
Objective: Compare models (XLM-Roberta, mBERT, DistilBERT) and pick the best.

Steps:
✅ Fine-Tune Multiple Models
✅ Evaluate Metrics (accuracy, inference speed)
✅ Select Best Model (balance of speed & performance)

📂 Output:

model_comparison_report.md

##### Task 5: Model Interpretability
Objective: Explain model predictions using SHAP/LIME.

Steps:
✅ Run SHAP/LIME on Sample Predictions
✅ Identify Weak Spots (e.g., misclassified prices)
✅ Generate Interpretability Report

📂 Output:

shap_analysis.html (interactive explanations)

##### Task 6: FinTech Vendor Scorecard
Objective: Rank vendors for micro-lending eligibility.

Steps:
✅ Calculate Metrics:

Posting Frequency (posts/week)

Avg. Views per Post

Avg. Product Price (ETB)

✅ Compute Lending Score

python
score = (avg_views * 0.5) + (post_freq * 0.3) + (avg_price * 0.2)
✅ Generate HTML Report

Leaderboard of top vendors.

Key metrics comparison.

📂 Output:

vendor_scorecard.html

#### How to Run the Project
1. Data Collection
python
python scraper.py --channels "channel1,channel2" --output raw_data.json
2. NER Training
python
python train_ner.py --data ner_labels.conll --model xlm-roberta-base
3. Vendor Analytics
python
python vendor_analytics.py --data processed_data.csv --model saved_model/
4. Generate Scorecard
python
python generate_scorecard.py --output report.html
Future Improvements
🔹 Expand NER to more languages (Oromo, Tigrinya).
🔹 Add image-based product recognition.
🔹 Deploy as a real-time API for lenders.
