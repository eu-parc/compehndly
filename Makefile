.PHONY: python-test r-setup r-setup-core r-setup-polars r-test

R_LIBS_USER ?= $(CURDIR)/.r-lib

python-test:
	cd python && uv run pytest -q

r-setup: r-setup-core r-setup-polars

r-setup-core:
	mkdir -p "$(R_LIBS_USER)"
	R_LIBS_USER="$(R_LIBS_USER)" Rscript -e 'install.packages(c("pkgload","testthat","jsonlite"), repos="https://cloud.r-project.org")'

r-setup-polars:
	mkdir -p "$(R_LIBS_USER)"
	R_LIBS_USER="$(R_LIBS_USER)" Rscript -e 'install.packages("polars", repos=c("https://pola-rs.r-universe.dev","https://cloud.r-project.org"))'

r-test:
	R_LIBS_USER="$(R_LIBS_USER)" Rscript R/tests/testthat.R
