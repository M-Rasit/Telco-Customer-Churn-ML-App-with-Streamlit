import pandas as pd
import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('log_model','rb'))

st.sidebar.title("Churn Probability of a Single Customer")

tenure = st.sidebar.slider("Tenure", 0,100,1)
totalcharges = st.sidebar.number_input("Total Charges")
contract = st.sidebar.selectbox("Contract", ["Month to month", "One year", "Two year"])
internetservice = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("OnlineSecurity", ["No", "No internet service", "Yes"])
techsupport = st.sidebar.selectbox("TechSupport", ["No", "No internet service", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
paymentmethod = st.sidebar.selectbox("PaymentMethod", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

df = pd.DataFrame(data = np.array([0]*14).reshape(1,14), columns=['tenure', 'TotalCharges', 'Contract_One year', 'Contract_Two year','InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
'TechSupport_No internet service', 'TechSupport_Yes', 'Dependents_Yes', 'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'])

if st.sidebar.button("Predict the churn"):
    df["tenure"] = tenure
    df["TotalCharges"] = totalcharges
    if contract == "One year":
        df["Contract_One year"] = 1 
    elif contract == "Two year":
        df["Contract_Two year"] = 1
    
    if internetservice == "Fiber optic":
        df["InternetService_Fiber optic"] = 1
    elif internetservice == "No":
        df["InternetService_No"] = 1
    
    if online_security == "No internet service":
        df["OnlineSecurity_No internet service"] == 1
    elif online_security == "Yes":
        df["OnlineSecurity_Yes"] = 1
    
    if techsupport == "No internet service":
        df["TechSupport_No internet service"] = 1
    elif techsupport == "Yes":
        df["TechSupport_Yes"] = 1
        
    if dependents == "Yes":
        df["Dependents_Yes"] = 1
        
    if paymentmethod == "Credit card (automatic)":
        df["PaymentMethod_Credit card (automatic)"] = 1
    elif paymentmethod == "Electronic check":
        df["PaymentMethod_Electronic check"] = 1
    elif paymentmethod == "Mailed check":
        df["PaymentMethod_Mailed check"] = 1
    
    pred = model.predict(df)    
    if pred[0] == 0:
        st.sidebar.success("This customer will stay with company!!!")
    elif pred[0] == 1:
        st.sidebar.error("This customer will churn!!!")


html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Churn Prediction ML App</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)        


new_df = pd.read_csv("random_df_telco_churn", index_col=0)

if st.checkbox("Churn Probability of Randomly Selected Customers"):
    
    st.markdown("### How many customers to be selected randomly?")
    number = st.selectbox("Please choose number of customers:", range(10,102,10))
    result_df = new_df.sample(number)
    selected_df = pd.get_dummies(result_df, drop_first=True)
    result_df["Churn Probability"] = [i[1] for i in model.predict_proba(selected_df)]
    st.table(result_df)
    
elif st.checkbox("Top Customers to Churn"):  
    
    st.markdown("### Top N Customers for Most Churn Probability")
    number = st.number_input("Please input number of customers:", min_value=1, step=1)
    total_df = new_df
    model_df = pd.get_dummies(total_df, drop_first=True)  
    total_df["Churn Probability"] = [i[1] for i in model.predict_proba(model_df)]
    total_df.sort_values(by="Churn Probability", ascending=False, inplace=True)
    st.table(total_df.head(number))
    
elif st.checkbox("Top Loyal Customers"):  
    
    st.markdown("### Top N Most Loyal Customers")
    number = st.number_input("Please input number of customers:", min_value=1, step=1)
    total_df = new_df
    model_df = pd.get_dummies(total_df, drop_first=True)  
    total_df["Churn Probability"] = [i[1] for i in model.predict_proba(model_df)]
    total_df.sort_values(by="Churn Probability", inplace=True)
    st.table(total_df.head(number))
       
    
