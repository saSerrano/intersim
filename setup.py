import setuptools 

with open("README.md", "r") as fh: 
	description = fh.read() 

setuptools.setup( 
	name="intersim", 
	version="1.0.0", 
	author="Sergio A. Serrano", 
	author_email="sserrano@inaoe.mx", 
	packages=["intersim"], 
	description="An implementation of the method proposed in the article titled 'Inter-task similarity measure for heterogeneous tasks'.", 
	long_description=description, 
	long_description_content_type="text/markdown", 
	license='MIT', 
	python_requires='>=3.8', 
	install_requires=['numpy',
				   'sklearn',
				   'matplotlib',
				   'mushroom_rl',
				   'json'
				   ]
) 