from setuptools import find_packages,setup
from typing import List
import os

HYPHEN_DOT_E = '-e .'
def get_requirement(file_path:str) -> List[str]:
    """
    This function read the requirements.txt file and return list of library which ned to be install
    """
    requirements = []
    if not os.path.exists(file_path):
        print("File not exists")
    else:
        with open(file_path,"r") as file_object:
            requirements=file_object.readlines()
            requirements=[i.replace('\n','')for i in requirements]
            print(requirements)
            if HYPHEN_DOT_E in requirements:
                requirements.remove(HYPHEN_DOT_E)
                print(requirements)
    return requirements
    
    
    
setup(
    name='mlproject01',
    version='0.0.1',
    author='Md Irshad',
    author_email='m3irshad3@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)