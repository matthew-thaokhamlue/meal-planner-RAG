from setuptools import setup, find_packages

setup(
    name="meal-shopping-assistant",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sentence-transformers>=2.2.2',
        'faiss-cpu>=1.7.4',
        'requests>=2.31.0',
        'python-dotenv>=1.0.0',
        'click>=8.1.7',
        'rich>=13.7.0',
        'pyyaml>=6.0.1',
        'pandas>=2.0.3',
        'python-dateutil>=2.8.2',
    ],
    entry_points={
        'console_scripts': [
            'meal-assistant=src.cli:main',
        ],
    },
    python_requires='>=3.8',
)

