from bokeh.models import Div


def get_layout(mode=None, **kwargs):
    desc = Div(text=f"""
    <p><i><b>
        Mode {mode} is not yet implemented.
        </br>
        Args: {str(kwargs)}
    </b></i></p>
    """)
    return desc
