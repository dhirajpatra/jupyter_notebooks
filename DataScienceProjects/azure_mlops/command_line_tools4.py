from setuptools import setup

setup(
  #...,
  entry_points = {
    'console_scripts': [
      'myscript = mypackage.mymodule:main_func',
    ]
  }
)