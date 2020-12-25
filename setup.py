import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name='jajucha',
    packages=setuptools.find_packages(),
    version='2.2.2',
    license='MIT',
    description='Controller Library for jajucha, a model car for autonomous driving education.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sanggyu Lee',
    author_email='sanggyu523@naver.com',
    url='https://github.com/gyusang/jajucha',
    download_url='https://github.com/gyusang/jajucha/archive/v_2_2_2.tar.gz',
    project_urls={
        'Source': 'https://github.com/gyusang/jajucha',
        'Report Bugs': 'https://github.com/gyusang/jajucha/issues'
    },
    keywords=['education', 'autonomous driving', 'jajucha', '자주차'],
    install_requires=[
        'numpy!=1.19.4',
        'opencv-python',
        'pyzmq',
        'imagezmq',
        'pillow',
        'scipy',
    ],
    python_requires='~=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    package_data={
        'jajucha': ['ABOUT.txt', 'CREDITS.txt'],
    }
)
