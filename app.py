import streamlit as st
import joblib
import re

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

MODEL_ACCURACY = 89.98


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    return text


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(text):

    cleaned_text = clean_text(text)

    text_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    negative_probability = probabilities[0] * 100
    positive_probability = probabilities[1] * 100

    if prediction == 1:
        sentiment = "Positive"
        confidence = positive_probability
    else:
        sentiment = "Negative"
        confidence = negative_probability

    return (
        sentiment,
        confidence,
        negative_probability,
        positive_probability
    )


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Overall application */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(168, 85, 247, 0.12),
                transparent 30%
            ),
            #080b14;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Main title */

    .main-title {
        text-align: center;
        font-size: 58px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Buttons */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-weight: 700;
        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );
        color: white;
    }

    .stButton > button:hover {
        box-shadow:
            0 8px 25px rgba(99, 102, 241, 0.35);
        transform: translateY(-1px);
    }

    /* Text area */

    textarea {
        border-radius: 14px !important;
    }

    /* Metric styling */

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.12);
        padding: 15px;
        border-radius: 15px;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 SentimentAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze movie reviews using Natural Language Processing '
    'and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns(
    [1.55, 1],
    gap="large"
)


# =========================================================
# LEFT SIDE — REVIEW ANALYZER
# =========================================================

with left:

    st.subheader("📝 Analyze a Review")

    st.caption(
        "Enter a movie review and let the trained machine "
        "learning model predict its sentiment."
    )

    review = st.text_area(
        "Movie Review",
        height=230,
        placeholder=(
            "Example:\n\n"
            "The movie was absolutely amazing! "
            "The acting was brilliant and the story "
            "kept me engaged from beginning to end."
        ),
        label_visibility="collapsed"
    )

    analyze_col, clear_col = st.columns(2)

    with analyze_col:

        analyze = st.button(
            "🔍 Analyze Sentiment"
        )

    with clear_col:

        clear = st.button(
            "🧹 Clear"
        )

    if clear:
        st.rerun()


# =========================================================
# RIGHT SIDE — DYNAMIC INSIGHTS
# =========================================================

with right:

    st.subheader("🎭 Sentiment Insights")

    st.caption(
        "Visualize what the model predicts from your review."
    )

    st.divider()

    # -----------------------------------------------------
    # BEFORE ANALYSIS
    # -----------------------------------------------------

    if not analyze:

        st.metric(
            "Model Accuracy",
            f"{MODEL_ACCURACY}%"
        )

        st.write("")

        st.write("🧠 **Machine Learning Model**")

        st.write(
            "Logistic Regression"
        )

        st.write("")

        st.write("🔤 **Feature Extraction**")

        st.write(
            "TF-IDF with Unigrams + Bigrams"
        )

        st.write("")

        st.write("🎬 **Training Dataset**")

        st.write(
            "IMDB Movie Reviews"
        )

        st.write("")

        st.write("📊 **Classification Task**")

        st.write(
            "Positive vs Negative"
        )

    # -----------------------------------------------------
    # AFTER ANALYSIS
    # -----------------------------------------------------

    else:

        if review.strip():

            (
                sentiment,
                confidence,
                negative_probability,
                positive_probability
            ) = predict_sentiment(review)

            st.write("### Model Prediction")

            if sentiment == "Positive":

                st.success(
                    f"😊 Positive Sentiment  •  {confidence:.2f}% confidence"
                )

            else:

                st.error(
                    f"😞 Negative Sentiment  •  {confidence:.2f}% confidence"
                )

            st.write("")

            st.write("**Positive Probability**")

            st.progress(
                int(round(positive_probability)),
                text=f"{positive_probability:.2f}%"
            )

            st.write("")

            st.write("**Negative Probability**")

            st.progress(
                int(round(negative_probability)),
                text=f"{negative_probability:.2f}%"
            )

            st.divider()

            st.write("### 🧠 Model Details")

            detail1, detail2 = st.columns(2)

            with detail1:

                st.metric(
                    "Accuracy",
                    f"{MODEL_ACCURACY}%"
                )

            with detail2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

        else:

            st.info(
                "👆 Enter a movie review on the left "
                "and click **Analyze Sentiment**."
            )


# =========================================================
# RESULT SECTION
# =========================================================

if analyze and review.strip():

    (
        sentiment,
        confidence,
        negative_probability,
        positive_probability
    ) = predict_sentiment(review)

    st.divider()

    if sentiment == "Positive":

        st.success(
            f"### 😊 Positive Review\n\n"
            f"The model predicts that this review has a "
            f"**positive sentiment** with **{confidence:.2f}% "
            f"model confidence**."
        )

    else:

        st.error(
            f"### 😞 Negative Review\n\n"
            f"The model predicts that this review has a "
            f"**negative sentiment** with **{confidence:.2f}% "
            f"model confidence**."
        )

    # =====================================================
    # REVIEW STATISTICS
    # =====================================================

    st.write("### 📊 Review Statistics")

    word_count = len(review.split())
    character_count = len(review)
    sentence_count = len(
        [s for s in re.split(r"[.!?]+", review) if s.strip()]
    )

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:

        st.metric(
            "Words",
            word_count
        )

    with stat2:

        st.metric(
            "Characters",
            character_count
        )

    with stat3:

        st.metric(
            "Sentences",
            sentence_count
        )

    with stat4:

        st.metric(
            "Accuracy",
            f"{MODEL_ACCURACY}%"
        )


# =========================================================
# EXAMPLE REVIEWS
# =========================================================

st.divider()

st.subheader("💡 Try an Example")

st.caption(
    "Use these sample reviews to test the sentiment analyzer."
)

example1, example2 = st.columns(2)

with example1:

    st.success(
        "😊 **Positive Example**\n\n"
        "An incredible movie with brilliant acting "
        "and a beautiful story. I absolutely loved it!"
    )

with example2:

    st.error(
        "😞 **Negative Example**\n\n"
        "A painfully boring movie with weak acting "
        "and a disappointing story."
    )


# =========================================================
# PROJECT FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Built with Python • Scikit-learn • TF-IDF • '
    'Logistic Regression • Streamlit'
    '</div>',
    unsafe_allow_html=True
)