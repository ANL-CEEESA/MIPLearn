project = "MIPLearn"
copyright = "2020-2021, UChicago Argonne, LLC"
author = ""
release = "0.2"
extensions = ["myst_parser", "nbsphinx"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_theme_options = {
    "repository_url": "https://github.com/ANL-CEEESA/MIPLearn/",
    "use_repository_button": False,
    "extra_navbar": "",
}
html_title = f"MIPLearn {release}"
nbsphinx_execute = 'never'
