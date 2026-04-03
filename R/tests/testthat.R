library(testthat)

# Load local package code directly for now.
pkgload::load_all(".", export_all = TRUE, helpers = FALSE, quiet = TRUE)

# Conformance is tested in:
#   R/tests/testthat/test-conformance.R
# and is included automatically by test_check().
test_check("compehndly")
