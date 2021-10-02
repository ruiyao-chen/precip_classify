init:
	conda env create --name "atmosphere" -f atmosphere.yml
test:
	nosetests tests
