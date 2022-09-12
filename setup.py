"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from setuptools import setup, find_packages

setup(
    name='ratingslib',
    version='1.0.0',
    description='Project for rating methods with applications',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ktalattinis/ratingslib',
    author='Kyriacos Talattinis',
    author_email='ktalattinis@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8'
    ],
    keywords=['ratings', 'ranking', 'machine learning', 'classification',
              'predictions', 'domain names', 'finance'],
    packages=find_packages(),
    package_data={"ratingslib": ["py.typed"]},
    include_package_data=True,
    python_requires='>=3.8',
    project_urls={
        'Source': 'https://github.com/ktalattinis/ratingslib',
    },
)
