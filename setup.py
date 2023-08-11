from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Neural Cellular Automata'
LONG_DESCRIPTION = 'Efficiently learn cellular automata rules that give desired emergent spatio-temporal patterning'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="NCA", 
        version=VERSION,
        author="Alex Richardson",
        author_email="<alexander.richardson@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
)