from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
	name='pyalign',
    version = '0.0.1',
    author = 'Jared Lumpe',
    author_email = 'mjlumpe@gmail.com',
    description = ('Fast C implementations of basic sequence alignment '
    	'algorithms wrapped in Python'),
    license='MIT',
    url='http://github.com/jlumpe/pyalign',
	packages=['pyalign'],
	ext_modules=[
		Extension(
			'pyalign.pwalign._pwalign',
			[
				'pyalign/pwalign/_pwalign/' + f
				for f in ['_pwalign.c', 'pwalign.c']
			],
			extra_compile_args=['-Werror', '-std=c99']
		)
	],
	include_dirs=get_numpy_include_dirs()
)
