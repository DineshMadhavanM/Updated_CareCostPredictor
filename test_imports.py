
import sys

print("Testing imports...")
try:
    import pandas
    print("pandas ok")
except ImportError as e:
    print(f"pandas failed: {e}")

try:
    import numpy
    print("numpy ok")
except ImportError as e:
    print(f"numpy failed: {e}")

try:
    import sklearn
    print("sklearn ok")
except ImportError as e:
    print(f"sklearn failed: {e}")

try:
    import xgboost
    print("xgboost ok")
except ImportError as e:
    print(f"xgboost failed: {e}")

try:
    import streamlit
    print("streamlit ok")
except ImportError as e:
    print(f"streamlit failed: {e}")

try:
    import plotly
    print("plotly ok")
except ImportError as e:
    print(f"plotly failed: {e}")

print("Done.")
