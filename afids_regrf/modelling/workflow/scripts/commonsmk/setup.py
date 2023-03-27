#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
	name = 'commonsmk',
	packages = find_packages(),
	version = '0.1',
	license='MIT',
	description = 'Helper for generating bids paths for snakemake workflows',
	author = 'Ali Khan',
	author_email = '',
	url = '',
	download_url = '',
	keywords = ['snakemake', 'bids', 'KEYWORDS'],
	classifiers=[
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3',
		'Operating System :: OS Independent',
	],
)


