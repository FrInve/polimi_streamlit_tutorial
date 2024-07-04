import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import io

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
    st.image("resources/dataset_card.png", width=400)

    st.write("This is the dataset we are using:")
    # This is called magic command and it will cache the output of the function
    load_diabetes(as_frame=True).frame

l_col, r_col = st.columns([0.4, 0.6])
with l_col:

    tab1, tab2 = st.tabs(["Input Features", "Select Line Number"])

    with tab1:
        with st.form("input_form"):
            st.write("Please input the features for the new data point:")
            # We have 10 features (input fields)
            age = st.selectbox("Age", options=list(range(0, 100)))
            sex = st.radio("Sex", options=["Male", "Female"])
            bmi = st.slider("BMI", min_value=10, max_value=40, value=24)
            bp = st.slider("BP", min_value=50, max_value=200, value=100, step=5)
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

            button_form_1 = st.form_submit_button("Submit New Point")
            # Scale the input features to the range of the dataset - very ugly, just for the sake of this example!
            new_point_1 = [
                ((age - 50) / 50) * 0.13,
                0.05 if sex == "Male" else -0.044,
                (bmi - 24) / 16 * 0.2,
                (bp - 50) / 150 * 0.15,
                (s1 - 250) / 250 * 0.2,
                (s2 - 250) / 250 * 0.2,
                (s3 - 250) / 250 * 0.2,
                (s4 - 250) / 250 * 0.2,
                (s5 - 250) / 250 * 0.2,
                (s6 - 250) / 250 * 0.2,
            ]

            new_point_1 = pd.DataFrame(
                [new_point_1], columns=load_diabetes().feature_names
            )

        st.bar_chart(new_point_1.T)

        if button_form_1:
            st.session_state["new_point"] = new_point_1

    with tab2:
        with st.form("line_number_form"):
            line_number = st.number_input(
                "Line Number", min_value=0, max_value=442, value=0
            )
            new_point_2 = load_diabetes(as_frame=True).data.iloc[line_number]
            new_point_2 = pd.DataFrame(
                [new_point_2], columns=load_diabetes().feature_names
            )
            button_form_2 = st.form_submit_button("Submit New Point")

        st.bar_chart(new_point_2.T)

        st.bar_chart(
            db.query(
                "SELECT pca1, pca2, pca3 FROM diabetes_reduced LIMIT 1 OFFSET "
                + str(line_number)
            ).T
        )

        if button_form_2:
            st.session_state["new_point"] = new_point_2

with r_col:
    if "last_prediction" in st.session_state:
        st.write(f"Last output was: {st.session_state['last_prediction']}")
    if st.button("Predict Severity of Diabetes"):

        prediction = classification.predict(st.session_state.new_point)
        regression = regression.predict(st.session_state.new_point)
        prediction_string = (
            f"Prediction: {prediction[0]} with a score of {regression[0]:.2f}"
        )
        st.session_state["last_prediction"] = prediction_string
        st.write(prediction_string)
        st.pyplot(make_3d_plot(st.session_state.new_point))

        download_btn_slot = st.empty()

        img = io.BytesIO()
        plt.savefig(img, format="png")

        download_btn = download_btn_slot.download_button(
            label="Download⬇", data=img, file_name="plot.png", mime="image/png"
        )
