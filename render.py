import PIL
from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output, display

def render_env(env, wait=True):
    """Render the environment using the PILSVG renderer

    NOTE: on WSL2, you have to have installed XLaunch and selected the option "Disable access control" to be able to render the environment.
    On top of that, you have to run the command "export DISPLAY=:0" on the terminal before running the code.

    Args:
        env (RailEnv): the flatland environment to render
        wait (bool, optional): whether to wait for the rendering to be displayed. Defaults to True.
    """

    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    clear_output(wait=wait)
    # display(pil_image)
    pil_image.show()