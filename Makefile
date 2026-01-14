
build:
	docker build -t linear-regression-ml .

run:
	docker run --rm linear-regression-ml  

clean:
	docker rmi linear-regression-ml