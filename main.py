import streamlit as st

st.title("Streamlit Title")

st.write("""
# Explore Different classifers
Which one is the best?
""")

dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))