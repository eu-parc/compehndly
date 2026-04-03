conformance_file_path <- function() {
  candidates <- c(
    file.path("shared", "conformance", "derived_variables_cases.json"),
    file.path("..", "shared", "conformance", "derived_variables_cases.json"),
    file.path("..", "..", "shared", "conformance", "derived_variables_cases.json")
  )

  for (p in candidates) {
    if (file.exists(p)) {
      return(normalizePath(p, mustWork = TRUE))
    }
  }

  normalizePath(candidates[[1]], mustWork = FALSE)
}

to_numeric_with_na <- function(x) {
  vapply(
    x,
    function(v) {
      if (is.null(v)) return(NA_real_)
      as.numeric(v)
    },
    numeric(1)
  )
}
