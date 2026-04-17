import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='llm-reasoning-mcts',
    description='LLM Reasoning with Monte Carlo Tree Search (MCTS)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='mcts, tree-search, large-language-models, llm, reasoning',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
