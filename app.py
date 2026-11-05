"""Diabetes Risk Screening Tool - Streamlit app (screening aid, NOT a diagnosis)."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load("model.joblib")


art = load_artifact()
model, features, thr, ref = art["model"], art["features"], art["threshold"], art["reference"]

YESNO = ["No", "Yes"]
AGE = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6, "50-54": 7,
       "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80 or older": 13}
GENHLTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
EDU = {"Never attended / kindergarten only": 1, "Elementary": 2, "Some high school": 3,
       "High school graduate": 4, "Some college / technical school": 5, "College graduate": 6}
INCOME = {"Less than $10,000": 1, "$10,000-$15,000": 2, "$15,000-$20,000": 3, "$20,000-$25,000": 4,
          "$25,000-$35,000": 5, "$35,000-$50,000": 6, "$50,000-$75,000": 7, "$75,000 or more": 8}
SEX = {"Female": 0, "Male": 1}

# feature -> (short name, question, kind, options)
META = {
    "HighBP": ("High blood pressure", "Has a doctor ever told you that you have high blood pressure?", "yesno", None),
    "HighChol": ("High cholesterol", "Has a doctor ever told you that you have high cholesterol?", "yesno", None),
    "CholCheck": ("Cholesterol check", "Have you had your cholesterol checked in the last 5 years?", "yesno", None),
    "BMI": ("BMI", "Your height and weight", "bmi", None),
    "Smoker": ("Smoking", "Have you smoked at least 100 cigarettes in your life?", "yesno", None),
    "Stroke": ("Stroke", "Have you ever had a stroke?", "yesno", None),
    "HeartDiseaseorAttack": ("Heart disease / attack", "Have you ever had coronary heart disease or a heart attack?", "yesno", None),
    "PhysActivity": ("Physical activity", "Did you do any physical activity or exercise in the past 30 days (not counting your job)?", "yesno", None),
    "Fruits": ("Fruit intake", "Do you eat fruit at least once a day?", "yesno", None),
    "Veggies": ("Vegetable intake", "Do you eat vegetables at least once a day?", "yesno", None),
    "HvyAlcoholConsump": ("Heavy drinking", "Are you a heavy drinker (men: 14+ drinks/week, women: 7+ drinks/week)?", "yesno", None),
    "AnyHealthcare": ("Health coverage", "Do you have any kind of health coverage?", "yesno", None),
    "NoDocbcCost": ("Skipped doctor (cost)", "In the past year, did you need a doctor but skip it because of cost?", "yesno", None),
    "GenHlth": ("General health", "How would you rate your general health?", "select", GENHLTH),
    "MentHlth": ("Poor mental-health days", "In the past 30 days, how many days was your mental health not good?", "days", None),
    "PhysHlth": ("Poor physical-health days", "In the past 30 days, how many days was your physical health not good?", "days", None),
    "DiffWalk": ("Difficulty walking", "Do you have serious difficulty walking or climbing stairs?", "yesno", None),
    "Sex": ("Sex", "Sex", "select", SEX),
    "Age": ("Age", "Age group", "select", AGE),
    "Education": ("Education", "Highest education level", "select", EDU),
    "Income": ("Income", "Annual household income", "select", INCOME),
}

st.title("🩺 Diabetes Risk Screening")
st.warning("**This is a screening aid, not a medical diagnosis.** Only a blood test and a doctor can tell "
           "whether you have diabetes or prediabetes.")
st.write(f"Answer {len(features)} quick questions to see if you should get a check-up.")

values = {}
with st.form("questionnaire"):
    for f in features:
        _, question, kind, opts = META[f]
        if kind == "yesno":
            values[f] = 1 if st.radio(question, YESNO, horizontal=True, key=f) == "Yes" else 0
        elif kind == "select":
            values[f] = opts[st.selectbox(question, list(opts.keys()), key=f)]
        elif kind == "days":
            values[f] = st.slider(question, 0, 30, 0, key=f)
        elif kind == "bmi":
            st.markdown(f"**{question}**")
            c1, c2 = st.columns(2)
            height = c1.number_input("Height (cm)", 100, 230, 170, key="height")
            weight = c2.number_input("Weight (kg)", 30, 250, 70, key="weight")
            values[f] = weight / ((height / 100) ** 2)
    submitted = st.form_submit_button("Check my risk", type="primary")

if submitted:
    try:
        x = pd.DataFrame([values])[features].astype(float)
        p = float(model.predict_proba(x)[0, 1])

        if p >= thr:
            level, box = "High", st.error
            msg = "Your answers are similar to many people who have diabetes or prediabetes. Please book a blood test (e.g. HbA1c or fasting glucose) with a doctor."
        elif p >= 0.5 * thr:
            level, box = "Moderate", st.warning
            msg = "Some of your answers raise your risk. Consider mentioning this at your next check-up."
        else:
            level, box = "Low", st.success
            msg = "Your answers look lower-risk. Keep up healthy habits and get regular check-ups."

        st.subheader("Your result")
        box(f"**{level} risk** - {msg}")
        st.metric("Estimated risk score", f"{p:.0%}")
        st.caption("The score is an estimate from survey data, not a probability that you have diabetes. "
                   "The 'High' level starts at a deliberately low cut-off so that most people who "
                   "have diabetes are flagged; many people flagged will not have it.")

        # What raises the risk? Compare with a typical person without diabetes (reference profile).
        rows = []
        for f in features:
            if abs(values[f] - ref[f]) > 1e-9:
                x2 = x.copy()
                x2[f] = ref[f]
                delta = p - float(model.predict_proba(x2)[0, 1])
                if delta > 0.002:
                    rows.append((META[f][0], delta * 100))
        if rows:
            st.subheader("What raises your score")
            fac = pd.DataFrame(rows, columns=["Factor", "Points added"]).sort_values("Points added", ascending=False)
            st.bar_chart(fac.set_index("Factor"))
            st.caption("Points added = how much lower your score would be if this answer matched a typical "
                       "person without diabetes. This shows association, not cause.")
        else:
            st.info("None of your answers push your score above a typical person without diabetes.")
    except Exception as e:  # never crash on bad input
        st.error(f"Something went wrong with your answers. Please check them and try again. ({e})")

with st.sidebar:
    st.header("About")
    st.write("Model trained on ~250K responses of the CDC BRFSS 2015 survey "
             "([UCI dataset](https://archive.ics.uci.edu/dataset/891)).")
    m = art.get("metrics", {})
    if m:
        st.write(f"Test recall: **{m['Recall']:.0%}** · Precision: **{m['Precision']:.0%}** · "
                 f"ROC-AUC: **{m['ROC-AUC']:.2f}**")
    st.caption("Limitations: self-reported US data from 2015; not validated for other populations; "
               "not a medical device.")
