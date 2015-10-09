from distutils.core import setup

setup(
    name='snowflake-openmp',
    version='0.1.0',
    url='github.com/ucb-sejits/snowflake-openmp',
    license='B',
    author='Nathan Zhang',
    author_email='nzhang32@berkeley.edu',
    description='OpenMP compiler for snowflake',

    packages=[
        'snowflake_openmp'
    ],

    install_requires=[
        'ctree',
        'snowflake'
    ]
)
