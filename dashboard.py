import plotly.express as px
import pandas as pd

st.markdown("## 📊 Model Performance")

data = {
    "Model": ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"],
    "Accuracy": [0.91, 0.88, 0.94, 0.93]
}

df = pd.DataFrame(data)
fig = px.bar(df, x="Model", y="Accuracy", title="Model Accuracy Comparison")
st.plotly_chart(fig)