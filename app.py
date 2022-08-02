import streamlit as st 
from fastai.vision.all import *
import pathlib 
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Transportni klasifikatsiya qiluvchi model')

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg'])
if file : 
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    # model
    model = load_learner('transform_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"To'g'ri kelgan transport turi : {pred}")
    st.info(f'Ehtimollik : {probs[pred_id]*100:.1f}%')
    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

