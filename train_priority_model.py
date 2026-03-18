"""
Train the maintenance request priority classifier.
Run this script once to generate priority_model.pkl and priority_vectorizer.pkl.
Re-run whenever you want to retrain on updated data.
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import os

BASE_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Seed dataset — (text, priority)
# text = title + " " + description + " " + category
# Expand this list as real labelled data accumulates in MongoDB.
# ---------------------------------------------------------------------------
SEED_DATA = [
    # ── HIGH ────────────────────────────────────────────────────────────────
    ("pipe burst flooding bathroom water everywhere Plumbing", "high"),
    ("major water leak ceiling dripping heavily Plumbing", "high"),
    ("no water supply entire unit dry taps 2 days Plumbing", "high"),
    ("sewage overflow bathroom smell unbearable Plumbing", "high"),
    ("toilet overflowing cannot flush sewage backing up Plumbing", "high"),
    ("sparks from electrical outlet fire hazard Electrical", "high"),
    ("electrical short circuit burning smell smoke Electrical", "high"),
    ("no electricity entire room power out Electrical", "high"),
    ("exposed live wire dangerous in bedroom Electrical", "high"),
    ("gas smell kitchen possible leak urgent Appliances", "high"),
    ("heater sparking burning smell Appliances", "high"),
    ("water heater burst flooding hot water Plumbing", "high"),
    ("main door lock broken cannot lock room security risk Carpentry", "high"),
    ("ceiling about to collapse cracks dangerous Carpentry", "high"),
    ("window glass shattered broken dangerous Carpentry", "high"),
    ("rat infestation multiple rodents health hazard Cleaning", "high"),
    ("cockroach infestation entire kitchen Cleaning", "high"),
    ("mold black toxic spreading all over wall Painting", "high"),
    ("AC leaking water electrical components sparking Appliances", "high"),
    ("fridge stopped working all food spoiling Appliances", "high"),
    ("no hot water system completely broken Plumbing", "high"),
    ("bathroom flooding water not draining emergency Plumbing", "high"),
    ("circuit breaker keeps tripping no power Electrical", "high"),
    ("door cannot open from inside stuck locked Carpentry", "high"),
    ("water damage ceiling stain bulging about to fall Plumbing", "high"),

    # ── MEDIUM ──────────────────────────────────────────────────────────────
    ("faucet dripping slowly water wasting Plumbing", "medium"),
    ("bathroom tap leaking minor drip Plumbing", "medium"),
    ("toilet running constantly water waste Plumbing", "medium"),
    ("low water pressure shower weak Plumbing", "medium"),
    ("kitchen sink draining slowly clogged Plumbing", "medium"),
    ("light bulb not working bedroom dark Electrical", "medium"),
    ("ceiling fan wobbling noisy Electrical", "medium"),
    ("power socket not working one outlet dead Electrical", "medium"),
    ("light switch flickering intermittent Electrical", "medium"),
    ("AC not cooling properly room warm Appliances", "medium"),
    ("washing machine not spinning properly Appliances", "medium"),
    ("oven not heating correctly Appliances", "medium"),
    ("door handle loose hard to open Carpentry", "medium"),
    ("wardrobe door off hinges Carpentry", "medium"),
    ("window latch broken difficult to close Carpentry", "medium"),
    ("cabinet hinge broken Carpentry", "medium"),
    ("room not cleaned properly dirty Cleaning", "medium"),
    ("bathroom tiles cracked chipped Carpentry", "medium"),
    ("kitchen exhaust fan not working Electrical", "medium"),
    ("intercom doorbell not working Electrical", "medium"),
    ("shower head clogged low pressure Plumbing", "medium"),
    ("bed frame broken wobbly Carpentry", "medium"),
    ("water stain on ceiling small Plumbing", "medium"),
    ("geyser not heating water adequately Appliances", "medium"),
    ("mosquito net window torn Carpentry", "medium"),

    # ── LOW ─────────────────────────────────────────────────────────────────
    ("wall needs repainting faded colour Painting", "low"),
    ("minor paint peeling small area Painting", "low"),
    ("scuff marks on wall cosmetic Painting", "low"),
    ("room general cleaning requested Cleaning", "low"),
    ("carpet minor stain small area Cleaning", "low"),
    ("window glass dirty needs cleaning Cleaning", "low"),
    ("bathroom deep cleaning required Cleaning", "low"),
    ("window screen small tear minor Carpentry", "low"),
    ("door squeaking needs oil minor Carpentry", "low"),
    ("shelf loose in wardrobe minor Carpentry", "low"),
    ("garden cleaning weeds minor outdoor Cleaning", "low"),
    ("ceiling fan dusty needs cleaning Cleaning", "low"),
    ("painting touch up small patch Painting", "low"),
    ("light switch cover cracked cosmetic Electrical", "low"),
    ("mirror has small crack cosmetic Carpentry", "low"),
    ("curtain rod slightly bent minor Carpentry", "low"),
    ("small crack in wall hairline cosmetic Painting", "low"),
    ("bathroom grout stained discoloured Cleaning", "low"),
    ("balcony sweeping cleaning required Cleaning", "low"),
    ("door stopper missing minor Carpentry", "low"),
    ("light shade dusty cosmetic Cleaning", "low"),
    ("minor rust on window frame Painting", "low"),
    ("small tile chip cosmetic bathroom Carpentry", "low"),
    ("AC filter needs cleaning dust Appliances", "low"),
    ("loose screw in door hinge minor Carpentry", "low"),
]

texts, labels = zip(*SEED_DATA)

# ---------------------------------------------------------------------------
# Build pipeline: TF-IDF (unigrams + bigrams) → Logistic Regression
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42
    ))
])

# Cross-validation to sanity-check accuracy
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
print(f"Cross-val accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")

# Train on full seed data
pipeline.fit(texts, labels)

# Save
joblib.dump(pipeline, os.path.join(BASE_DIR, "priority_model.pkl"))
print("Saved priority_model.pkl")

# Quick smoke test
test_cases = [
    ("pipe burst water flooding everywhere Plumbing", "high"),
    ("tap dripping slowly Plumbing", "medium"),
    ("wall needs repainting Painting", "low"),
]
print("\nSmoke tests:")
for text, expected in test_cases:
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    conf = dict(zip(classes, proba))
    status = "PASS" if pred == expected else "FAIL"
    print(f"  [{status}] '{text[:40]}' -> {pred} (expected {expected})")
