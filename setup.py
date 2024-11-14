import pip
from setuptools import setup, find_packages

setup(
    name="model_topic_divergence",
    version="0.1.0",
    description="A short description of your project",
    author="Adam King",
    author_email="aking@gumgum.com",
    url="https://github.com/gg-aking/model_topic_divergence",
    packages=find_packages(),  
    install_requires=pip.req.parse_requirements('requirements.txt'),
    python_requires=">=3.6",
)