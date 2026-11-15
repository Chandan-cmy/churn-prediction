import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title='Churn Predictor',
    page_icon='📊',
    layout='wide'
)

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

model = load_model()

st.title('📊 Telco Customer Churn Predictor')
st.markdown('*Predict whether a customer is likely to churn using ML.*')
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────
st.sidebar.header('Customer Details')

tenure         = st.sidebar.slider('Tenure (months)', 0, 72, 12)
monthly_charge = st.sidebar.slider('Monthly Charges ($)', 18, 120, 65)
total_charges  = st.sidebar.number_input('Total Charges ($)', 0.0, 9000.0, float(monthly_charge * tenure))
contract       = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet       = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment        = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
gender         = st.sidebar.selectbox('Gender', ['Male', 'Female'])
senior         = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
partner        = st.sidebar.selectbox('Has Partner?', ['Yes', 'No'])
dependents     = st.sidebar.selectbox('Has Dependents?', ['No', 'Yes'])

# ── Build input row ───────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'tenure':          tenure,
    'MonthlyCharges':  monthly_charge,
    'TotalCharges':    total_charges,
    'Contract':        contract,
    'InternetService': internet,
    'PaymentMethod':   payment,
    'gender':          gender,
    'SeniorCitizen':   1 if senior == 'Yes' else 0,
    'Partner':         partner,
    'Dependents':      dependents,
    'PhoneService':        'Yes',
    'MultipleLines':       'No',
    'OnlineSecurity':      'No',
    'OnlineBackup':        'No',
    'DeviceProtection':    'No',
    'TechSupport':         'No',
    'StreamingTV':         'No',
    'StreamingMovies':     'No',
    'PaperlessBilling':    'Yes',
}])

# ── Predict ───────────────────────────────────────────────────────────────
THRESHOLD = 0.343   # tuned threshold from notebook

if st.sidebar.button('Predict Churn 📊', use_container_width=True):
    probability = model.predict_proba(input_data)[0][1]
    prediction  = int(probability >= THRESHOLD)

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error('⚠️ HIGH CHURN RISK')
        else:
            st.success('✅ LOW CHURN RISK')
        st.metric('Churn Probability', f'{probability:.1%}')

    with col2:
        st.info('**Recommended Action**')
        if probability > 0.7:
            st.write('🔴 Offer retention discount immediately.')
        elif probability > 0.4:
            st.write('🟡 Flag for proactive outreach next month.')
        else:
            st.write('🟢 No immediate intervention needed.')

    # Show what drove this prediction
    st.divider()
    st.subheader('Key risk factors for this customer')
    factors = []
    if contract == 'Month-to-month':
        factors.append('⚠️ Month-to-month contract — highest churn risk contract type')
    if tenure < 12:
        factors.append(f'⚠️ Low tenure ({tenure} months) — new customers churn more')
    if internet == 'Fiber optic':
        factors.append('⚠️ Fiber optic service — higher churn segment')
    if monthly_charge > 70:
        factors.append(f'⚠️ High monthly charges (${monthly_charge}) — price sensitivity risk')
    if factors:
        for f in factors:
            st.write(f)
    else:
        st.write('✅ No major risk factors detected.')

st.divider()
st.caption('Model: Gradient Boosting Classifier | Tuned threshold: 0.343 | Trained on IBM Telco Churn dataset')