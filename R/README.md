# compehndly (R)

This folder contains the R implementation of derived-variable functions, aligned
with the Python behavior through shared conformance vectors.

## Prerequisites

Install core R tooling:

```r
install.packages(c("pkgload", "testthat", "jsonlite"))
```

Install `polars` for R (required by `compehndly_apply`) from r-universe:

```r
install.packages("polars", repos = c("https://pola-rs.r-universe.dev", "https://cloud.r-project.org"))
```

On Ubuntu/Debian, if compiling dependencies fails (for example `fs`), install build tools first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libcurl4-openssl-dev libssl-dev libxml2-dev
```

## Load The Local Package

From the repository root:

```r
pkgload::load_all("R", export_all = TRUE)
```

This loads the functions defined in:

- `R/R/derived_variables.R`

## Run Derived Variables

Main entrypoint:

- `compehndly_apply(function_name, ..., .params = list())`

Example:

```r
library(polars)

measured <- pl$Series(c(50.0, 100.0, 75.0))
sg <- pl$Series(c(1.020, 1.015, 1.025))

out <- compehndly_apply(
  "normalize_specific_gravity",
  measured = measured,
  sg_measured = sg,
  .params = list(sg_ref = 1.024)
)

out$to_list()
```

## Run Shared Conformance Tests

Shared vectors live in:

- `shared/conformance/derived_variables_cases.json`

Run R tests from repository root:

```r
testthat::test_dir("R/tests/testthat")
```

Conformance scaffold:

- `R/tests/testthat/test-conformance.R`
