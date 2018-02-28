from distutils.core import setup
setup(name='oplrareg',
      description='Mixed Integer Piecewise Linear Regression with Regularisation',
      version='0.1',
      py_modules=['oplrareg'],
      author=['Jonathan Cardoso Silva'],
      author_email=['jonathan.car.silva@gmail.com'],
      url=['https://github.com/KISysBio/OPLRAreg'],
      url_download=['https://github.com/KISysBio/OPLRAreg/tarball/master'],
      keywords=[''],
      install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'pyomo'],
      entry_points={
        'console_scripts': [
            'oplrareg = oplrareg.__main__:main'
        ]
      }
    )
