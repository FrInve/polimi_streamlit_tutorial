import streamlit as st
import numpy as np
import pandas as pd
from streamlit.components.v1 import html
import base64


# Config Page
st.set_page_config(
    page_title="Home Page",
    # layout = "wide",
    # initial_sidebar_state="collapsed"
)


# Header of streamlit
st.header("Home Page")


# Customized header with markdown + html/css

st.markdown(
    """
    <style>
        div[data-testid="stMarkdownContainer"]
        {
            text-align: center;
            align-items: center;
        }
    </style>    
    """,unsafe_allow_html=True
)

st.markdown("""
            <style>
        h1 {
            font-family: Trebuchet MS, sans-serif;
            font-size: 45px;
            font-weight: bold;
            text-align: center;
        }
    </style>""", unsafe_allow_html=True)
st.markdown("""<h1>Home Page</h1>""", unsafe_allow_html=True)



# Ways to move around between pages
st.write("First method to change page:")
st.page_link("pages/1_Analisi.py", use_container_width = True) 
#... but it is quite restricted in functionalities... let's try with markdown
st.write("Second method to change page:")
st.markdown("""<a class="button-change" href="/Analisi">Analisi</a>""", unsafe_allow_html=True)

#... but it opens a new page -> we want to avoid it... but we need javascript
js = '''
<script>
    var navigationLinks = window.parent.document.getElementsByClassName("button-change");
    var cleanNavbar = function(navigation_element) {
        navigation_element.removeAttribute('target')}
    for (var i = 0; i < navigationLinks.length; i++) {
            cleanNavbar(navigationLinks[i]);}
</script>
'''

# Why introducing this way of link to other pages? For query parameters!

col1, col2 = st.columns([1,1])

with col1:
    with st.container(border = True):
        st.markdown("""<a class="button-change" href="/Analisi/?lang=english">English Version</a>""", unsafe_allow_html=True)
with col2:
    with st.container(border = True):
        st.markdown("""<a class="button-change" href="/Analisi/?lang=italian">Italian Version</a>""", unsafe_allow_html=True)


# here just for blank space generation
html(js)




