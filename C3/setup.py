from setuptools import setup, find_packages

setup(
    name='c3_scheduler',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'onnx>=1.15.0',
        'cupy-cuda12x>=13.0.0',
        'scipy>=1.10.0',
        'tabulate>=0.9.0',
    ],
    entry_points={
        'console_scripts': [
            'c3-export-dag=cli.export_dag:main',
            'c3-infer=cli.infer:main',
        ],
    },
)
