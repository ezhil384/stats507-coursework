# YouTube Sentiment Analysis Project

This project explores the relationship between the sentiment of YouTube video comments and video popularity metrics such as likes and views. Using both traditional machine learning models and state-of-the-art transformer-based NLP models, we analyze sentiment from top comments and assess its influence on engagement.

## 📁 Dataset
The dataset consists of two CSV files:
- `videos-stats.csv`: Contains metadata about each video including views, likes, title, and comment count.
- `comments.csv`: Contains the top 10 most relevant comments for each video, their number of likes, and a labeled sentiment score (0: negative, 1: neutral, 2: positive).

## 💡 Objective
1. Predict sentiment from comments using both traditional ML and Hugging Face transformer models.
2. Analyze the correlation between average comment sentiment and video engagement (likes/views).

## 🧰 Tools & Technologies
- Python, Pandas, Scikit-learn
- Hugging Face Transformers
- RoBERTa (fine-tuned)
- BART (zero-shot classification)
- Seaborn & Matplotlib for visualizations

## 🧠 Models Used
- TF-IDF + Logistic Regression
- TF-IDF + Support Vector Machine (SVM)
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + Random Forest
- Transformer (RoBERTa)
- Zero-shot Classification (BART)

## 📈 Evaluation Metrics
All models are evaluated using:
- Accuracy
- F1 Score (weighted)
- Precision & Recall (in full reports)

## 📊 Key Visualizations
- Scatter plots showing average sentiment vs views/likes
- Bar chart comparing model performance (accuracy & F1 score)

## 📁 Outputs
- `model_comparison_scores.png`: Bar chart comparing models
- `avg_sentiment_vs_views.png`, `avg_sentiment_vs_likes.png`: Visual analysis of sentiment vs engagement
- `youtube_sentiment_results.csv`: Annotated dataset with model predictions

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies:
```bash
pip install pandas scikit-learn transformers datasets matplotlib seaborn
```
3. Run the Jupyter Notebook or script provided to reproduce the results.

## 📄 Report
The project summary and analysis are compiled in a 2-page IEEE conference paper format located in `report.pdf` or `main.tex`.

## ✍️ Author
**Abhiraj Ezhil**  
Department of Statistics, University of Michigan

---
Feel free to fork, improve or use this analysis for academic/research purposes!
