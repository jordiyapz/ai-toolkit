from setuptools import setup, find_packages

setup(
    name='jordiyapz-ai-toolkit',
    version='1.0.0',
    url='https://github.com/jordiyapz/ai-toolkit.git',
    author='Jordi Yaputra',
    author_email='jordiyaputra@gmail.com',
    description='Toolkit yang terkumpul dari tubes AI pertama (Genetic Algorithm) hingga tubes pertama ML (Unsupervised Learning).',
    packages=find_packages(),
    install_requires=['pandas >= 1.2.4'],
)