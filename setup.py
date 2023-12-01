from setuptools import setup, find_packages

setup(
    name='Cochl_CAM',
    version='1.0',
    author='Jaeyeong Hwang',
    author_email='jyhwang@cochlear.ai',
    description='The Grad-CAM test script about Models',
    url='https://github.com/jwyeeh-dev/Cochl_CAM',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tf-keras-vis',
        'tensorflow',
        'keras',
        'tf-keras-vis',
        'tf-explain'
    ],
    entry_points={
        'console_scripts': [
            'my_script = my_package.module:main_function',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
