import setuptools

NAME = "danna_sep"
VERSION = '1.0'
AUTHOR = 'Chin-Yun Yu'
EMAIL = 'lolimaster.cs03@nctu.edu.tw'

packages = setuptools.find_packages()

package_data = {'': ['*.pth']}

install_requires = ['torch', 'torchaudio', 'gdown', 'norbert']

entry_points = {'console_scripts': ['danna_sep = danna_sep:entry']}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_kwargs = {
    'name': NAME,
    'version': VERSION,
    'author': AUTHOR,
    'author_email': EMAIL,
    'description': '',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/yoyololicon/danna-sep',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'classifiers': [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
}


setuptools.setup(**setup_kwargs)
