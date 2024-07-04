import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("An Ordinary Dashboard")
st.write("Welcome to an Ordinary Dashboard! 🥳")

##### BACKEND #####
# These are our models and behave as backend entities!
classification = joblib.load("models/classification.pkl")
regression = joblib.load("models/regression.pkl")
pca = joblib.load("models/pca.pkl")
df_reduced = pd.read_csv("data/df_reduced.csv")
db = st.connection(name="diabetes", type="sql", url="sqlite:///data/diabetes.db")


def make_3d_plot(new_point):

    new_point_reduced = pca.transform(new_point)
    new_point_reduced = pd.DataFrame(
        new_point_reduced, columns=["pca1", "pca2", "pca3"]
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        df_reduced.pca1,
        df_reduced.pca2,
        df_reduced.pca3,
        c=df_reduced.target_class,
        cmap="coolwarm",
    )
    ax.scatter(
        new_point_reduced.pca1,
        new_point_reduced.pca2,
        new_point_reduced.pca3,
        c="black",
        marker="P",
        s=200,
    )
    ax.set_xlabel("pca1")
    ax.set_ylabel("pca2")
    ax.set_zlabel("pca3")

    return fig


##### FRONTEND #####

expander = st.expander("Dataset")
with expander:
    st.image("resources/dataset_card.png", use_column_width=True)

    st.write("This is the dataset we are using:")
    # This is called magic command and it will cache the output of the function
    load_diabetes(as_frame=True).frame

l_col, r_col = st.columns([0.3, 0.7])
with l_col:

    tab1, tab2 = st.tabs(["Input Features", "Select Line Number"])

    with tab1:
        with st.form("input_form"):
            st.write("Please input the features for the new data point:")
            # We have 10 features (input fields)
            age = st.number_input("Age", min_value=0, max_value=150, value=50)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            bmi = st.number_input("BMI", min_value=0, max_value=100, value=25)
            bp = st.slider("BP", min_value=0, max_value=200, value=100, step=5)
            s1 = st.number_input(
                "Total Serum Cholesterol", min_value=0, max_value=500, value=100
            )
            s2 = st.number_input("LDL", min_value=0, max_value=500, value=100)
            s3 = st.number_input("HDL", min_value=0, max_value=500, value=100)
            s4 = st.number_input(
                "Total Cholesterol", min_value=0, max_value=500, value=100
            )
            s5 = st.number_input(
                "Log Serum Trig", min_value=0, max_value=500, value=100
            )
            s6 = st.number_input("Blood Sugar", min_value=0, max_value=500, value=100)

            # We have 2 models, one for classification and one for regression
            new_point = [
                age - 50 / 100,
                0.05 if sex == "Male" else -0.044,
                bmi - 50 / 50,
                bp - 100 / 100,
                s1 - 250 / 250,
                s2 - 250 / 250,
                s3 - 250 / 250,
                s4 - 250 / 250,
                s5 - 250 / 250,
                s6 - 250 / 250,
            ]
            new_point = pd.DataFrame([new_point], columns=load_diabetes().feature_names)
            st.form_submit_button("Load New Point")

        st.bar_chart(new_point.T)

    with tab2:
        line_number = st.number_input(
            "Line Number", min_value=0, max_value=442, value=0
        )
        new_point = load_diabetes(as_frame=True).data.iloc[line_number]
        new_point = pd.DataFrame([new_point], columns=load_diabetes().feature_names)

        st.bar_chart(new_point.T)

        st.bar_chart(
            db.query(
                "SELECT pca1, pca2, pca3 FROM diabetes_reduced LIMIT 1 OFFSET "
                + str(line_number)
            ).T
        )

with r_col:
    if st.button("Predict Severity of Diabetes"):
        prediction = classification.predict(new_point)
        regression = regression.predict(new_point)
        st.write(f"Prediction: {prediction[0]} with a score of {regression[0]:.2f}")
        st.pyplot(make_3d_plot(new_point))
