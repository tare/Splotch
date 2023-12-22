#!/usr/bin/env python

import sys
import os
import glob

from distutils.core import setup
from distutils.command.install_data import install_data
from setuptools.command.install import install

import urllib.request
import re
import tarfile
import tempfile

import splotch

import subprocess

def compile_stan_code():
    cmdstan_latest_url = 'https://api.github.com/repos/stan-dev/cmdstan/releases/latest'

    # try to get the latest version number of CmdStan
    req = urllib.request.Request(cmdstan_latest_url)
    with urllib.request.urlopen(req) as response:
        content = response.read()
        latest_version = re.search('"tag_name":"([^"]*)"',str(content)).group(1)
        
    # download the latest version of CmdStan
    url_template = 'https://github.com/stan-dev/cmdstan/releases/download/%s/cmdstan-%s.tar.gz'
    filename,_ = urllib.request.urlretrieve(url_template%(latest_version,latest_version.replace('v','')))

    # extract the source file
    temp_dir = tempfile.TemporaryDirectory()
    with tarfile.open(filename) as f:
        f.extractall(temp_dir.name)

    # compile CmdStan
    proc = subprocess.run(['make','build'],cwd=os.path.join(temp_dir.name,'cmdstan-%s'%(latest_version.replace('v',''))),stdout=sys.stdout,stderr=sys.stderr)
    if proc.returncode:
        if proc.stderr:
            print(proc.stderr.decode('utf-8').strip())
        sys.exit(3)

    # find our stan models (i.e. stan/*.stan)
    stan_files = glob.glob(os.path.join(os.getcwd(),'stan','*.stan'))
    # compile our stan models
    for stan_file in stan_files:
        proc = subprocess.run(['make',re.sub('.stan$','',stan_file)],cwd=os.path.join(temp_dir.name,'cmdstan-%s'%(latest_version.replace('v',''))),stdout=sys.stdout,stderr=sys.stderr)
        if proc.returncode:
            print('Command "make %s" failed'%(re.sub('.stan$','',stan_file)))
            if proc.stderr:
                print(proc.stderr.decode('utf-8').strip())
            sys.exit(3)

# our custom install_data script
# downloads and compiles CmdStan
# and compiles our Stan models
class StanCodeCompilation(install_data):
    def run(self):

        # let's not continue if the user didn't want to install CmdStan
        # and compile the Stan models
        if not compile_stan:
            return

        # compile our stan code first
        compile_stan_code()
        # run the default install_data script
        install_data.run(self)

# our custom install script
# captures the install_option --stan
class InstallCommand(install):
    user_options = install.user_options + [('stan',None,None)]

    def initialize_options(self):
        install.initialize_options(self)
        self.stan = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        #print(self.stan)
        global compile_stan

        if self.stan:
            compile_stan = True
        else:
            compile_stan = False

        install.run(self)

# read the long description
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'README.md'),encoding='utf-8') as f:
    long_description = f.read()

# read the package requirements
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'requirements.txt'),encoding='utf-8') as f:
    install_requires = f.read().splitlines()

# get the names of thestan executables
stan_files = [re.sub('.stan$','',stan_file) for stan_file in glob.glob(os.path.join(os.getcwd(),'stan','*.stan'))]

setup(name='Splotch',
      version=splotch.__version__,
      description='Hierarchical model for Spatial Transcriptomics data',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=splotch.__author__,
      author_email=splotch.__email__,
      url='https://github.com/tare/Splotch',
      license=splotch.__license__,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD 3-Clause "New" or "Revised" License (BSD-3-Clause)',
          'Programming Language :: Python :: 3'],
      packages=['splotch'],
      scripts=['bin/splotch_prepare_count_files','bin/splotch_generate_input_files','bin/splotch'],
      # could not pass binaries through the scripts argument
      data_files=[('bin',stan_files)],
      install_requires=install_requires,
      cmdclass={
          'install': InstallCommand,
          'install_data': StanCodeCompilation}
)
