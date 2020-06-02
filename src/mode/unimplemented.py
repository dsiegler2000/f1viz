from bokeh.models import Div
from urllib.parse import quote


def get_layout(mode=None, **kwargs):
    error_text = quote("Please include this information: " + str(kwargs))
    desc = Div(text=f"""
    <p><i><b>
        Mode {mode} is not yet implemented or we've encountered an error. Please email us: 
        <a href="mailto:f1vizwebmaster@gmail.com?subject=Bug%20Report&body={error_text}">
        f1vizwebmaster@gmail.com</a>
        </br>
        Args: {str(kwargs)}
    </b></i></p>
    """)
    return desc
