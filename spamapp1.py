import streamlit as st
import pickle
import os

# Get the current directory (where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the vectorizer and model using relative paths
with open(os.path.join(BASE_DIR, "vec123.pkl"), "rb") as f:
    cv = pickle.load(f)

with open(os.path.join(BASE_DIR, "spam123.pkl"), "rb") as f:
    model = pickle.load(f)


def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    
    st.subheader("Classification")
    user_input = st.text_area("Enter an email to classify", height=150)
    
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.write("Please enter an email to classify.")


if __name__ == "__main__":
    main()


